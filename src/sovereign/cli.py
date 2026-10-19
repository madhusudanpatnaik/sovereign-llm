"""`sovereign` — fine-tune, quantize, serve and measure a small open-weight model against a frontier
API on PubMedQA.

    sovereign download --list                        # what this project downloads, and how big
    sovereign download                               # fetch the model + PubMedQA (on the GPU box)
    sovereign data                                   # splits + label balance (downloads ~3 MB)
    sovereign memory --model Qwen/Qwen3-4B-Instruct-2507 --gpu-gib 24   # config.json only
    sovereign smoke                                  # CPU: tiny model, real LoRA, eval loop
  on a GPU box:
    sovereign train --precision nf4 --output-dir runs/qlora
    sovereign serve --adapter pubmedqa=runs/qlora [--precision nf4] [--no-prefix-caching] --exec
    sovereign merge runs/qlora runs/merged && sovereign compress runs/merged runs/merged-fp8
    sovereign serve --model runs/merged-fp8 --precision fp8 --exec    # the throughput arm
    sovereign eval --base-url http://127.0.0.1:8000/v1 --model pubmedqa --out preds/qlora.jsonl
    sovereign eval ... --constrained                 # replies decoded as exactly yes/no/maybe
    sovereign eval --anthropic --out preds/claude.jsonl          # needs ANTHROPIC_API_KEY
    sovereign compare preds/*.jsonl --reference claude
    sovereign gate --candidate preds/qlora-nf4.jsonl --baseline preds/qlora-bf16.jsonl
    sovereign bench --base-url ... --model pubmedqa --concurrency 1,4,16,64 [--workload few-shot]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sovereign import data as pubmedqa
from sovereign.bench import RunStats, closed_loop, few_shot, open_loop, zero_shot
from sovereign.clients import (
    AnthropicClient,
    ChatClient,
    OpenAICompatibleClient,
    choice_constraint,
)
from sovereign.evaluate import (
    paired_bootstrap,
    predict,
    read_predictions,
    score,
    write_predictions,
)
from sovereign.gate import GatePolicy, decide


def _splits() -> pubmedqa.Splits:
    return pubmedqa.split(pubmedqa.load_labeled())


def _cmd_download(args: argparse.Namespace) -> int:
    from sovereign import assets

    chosen = assets.find(args.names)
    if args.list or args.dry_run:
        print("\n".join(assets.describe(chosen)))
        if args.dry_run:
            print("\n(dry run: nothing downloaded)")
        return 0
    for line in assets.download(chosen, cache_dir=args.cache_dir):
        print(line)
    return 0


def _cmd_data(_: argparse.Namespace) -> int:
    s = _splits()
    for name in ("train", "val", "test"):
        rows = getattr(s, name)
        counts = pubmedqa.label_counts(rows)
        share = "  ".join(f"{k}={v} ({v / len(rows):.1%})" for k, v in counts.items())
        print(f"{name:>5}: {len(rows):4d}  {share}")
    majority = max(pubmedqa.label_counts(s.test).values()) / len(s.test)
    print(f"majority-class accuracy on test: {majority:.1%} — the floor any model must beat")
    return 0


def _cmd_memory(args: argparse.Namespace) -> int:
    from sovereign import quant

    if args.config:
        from transformers import AutoConfig

        mem = quant.from_config(AutoConfig.from_pretrained(args.config))
    else:
        mem = quant.from_hub(args.model)
    p = mem.params
    print(
        f"{args.config or args.model}: {p.total / 1e9:.3f}B params "
        f"({p.linear_weights / p.total:.1%} in quantizable linear layers)"
    )
    print(f"KV cache: {mem.kv_bytes_per_token / 1024:.0f} KiB per token (bf16)")
    print(
        f"\n{'precision':>9} {'weights':>10} {'KV budget':>10} {'tokens':>9} "
        f"{f'seqs@{args.context}':>10}"
    )
    for prec in quant.PRECISIONS:
        c = quant.capacity(mem, prec, gpu_gib=args.gpu_gib, context_tokens=args.context)
        print(
            f"{prec:>9} {c.weights_gib:>8.2f}Gi {c.kv_budget_gib:>8.2f}Gi "
            f"{c.tokens_in_flight:>9,} {c.sequences_at_context:>10}"
        )
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from sovereign import smoke

    r = smoke.run()
    print(f"LoRA trainable params: {r.trainable_params:,} of {r.total_params:,}")
    print(
        f"loss {r.first_loss:.3f} -> {r.last_loss:.3f}; base weights unchanged: "
        f"{r.base_weights_unchanged}; adapter weights changed: {r.adapter_weights_changed}"
    )
    s = r.scores
    print(f"held-out toy accuracy {s.accuracy:.1%} (invalid {s.invalid_rate:.1%}, n={s.n})")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from sovereign.train import TrainConfig, train

    s = _splits()
    cfg = TrainConfig(
        model_id=args.model,
        precision=args.precision,
        output_dir=args.output_dir,
        epochs=args.epochs,
        r=args.r,
        alpha=args.alpha,
    )
    print(json.dumps(train(cfg, s.train, s.val), indent=2))
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    from sovereign.train import merge

    merge(args.model, args.adapter_dir, args.out_dir)
    return 0


def _cmd_compress(args: argparse.Namespace) -> int:
    from sovereign.compress import CompressConfig, compress

    cfg = CompressConfig(
        model_dir=args.model_dir,
        out_dir=args.out_dir,
        scheme=args.scheme,
        num_calibration_samples=args.calibration_samples,
    )
    train = _splits().train if args.scheme == "w4a16" else []
    print(f"wrote {compress(cfg, train)}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import os

    from sovereign.compress import is_compressed
    from sovereign.serve import ServeConfig, command, shell

    if args.precision in ("fp8", "w4a16") and not is_compressed(args.model):
        raise SystemExit(
            f"--precision {args.precision} serves a checkpoint from `sovereign compress`; "
            f"{args.model} has no quantization config, so vLLM would serve it unquantized"
        )

    pairs = [a.split("=", 1) for a in args.adapter]
    if any(len(p) != 2 or not all(p) for p in pairs):
        raise SystemExit(f"--adapter takes name=path, e.g. pubmedqa=runs/qlora; got {args.adapter}")
    adapters = {name: path for name, path in pairs}
    cfg = ServeConfig(
        model=args.model,
        precision=args.precision,
        port=args.port,
        prefix_caching=not args.no_prefix_caching,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        lora_adapters=adapters,
    )
    print(shell(cfg))
    if args.exec:
        import shutil

        if shutil.which("vllm") is None:
            raise SystemExit(
                "vllm is not installed: pip install -r requirements-serve.txt (Linux with a CUDA "
                "GPU; there are no macOS wheels)"
            )
        cmd = command(cfg)
        os.execvp(cmd[0], cmd)
    return 0


def _client(args: argparse.Namespace) -> ChatClient:
    if args.anthropic:
        return AnthropicClient(effort=args.effort)
    if not (args.base_url and args.model):
        raise SystemExit(
            "give --base-url and --model (an OpenAI-compatible server), or --anthropic"
        )
    extra = choice_constraint(pubmedqa.LABELS) if args.constrained else None
    return OpenAICompatibleClient(args.base_url, args.model, extra_body=extra)


def _cmd_eval(args: argparse.Namespace) -> int:
    test = _splits().test[: args.limit or None]
    client = _client(args)
    preds = asyncio.run(predict(client, test, concurrency=args.concurrency))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_predictions(args.out, preds)
    s = score(client.name, preds)
    print(
        f"{client.name}: accuracy {s.accuracy:.1%}  macro-F1 {s.macro_f1:.3f}  "
        f"invalid {s.invalid_rate:.1%}  (n={s.n}, majority baseline {s.majority_baseline:.1%})"
    )
    return 0


def _name(path: str) -> str:
    return Path(path).stem


def _cmd_compare(args: argparse.Namespace) -> int:
    runs = {_name(p): read_predictions(p) for p in args.predictions}
    print(f"{'system':<24}{'n':>5}{'acc':>8}{'macro-F1':>10}{'invalid':>9}{'p95 s':>8}")
    for name, preds in runs.items():
        s = score(name, preds)
        print(
            f"{name:<24}{s.n:>5}{s.accuracy:>8.1%}{s.macro_f1:>10.3f}{s.invalid_rate:>9.1%}"
            f"{s.p95_latency_s:>8.2f}"
        )
    if args.reference:
        if args.reference not in runs:
            raise SystemExit(f"--reference {args.reference!r} is not one of {sorted(runs)}")
        print(f"\naccuracy vs {args.reference} (paired bootstrap, 95% CI):")
        for name, preds in runs.items():
            if name == args.reference:
                continue
            d = paired_bootstrap(name, preds, args.reference, runs[args.reference])
            print(
                f"  {name:<22}{d.delta:+.1%}  [{d.ci_low:+.1%}, {d.ci_high:+.1%}]  "
                f"(only {name} right: {d.candidate_only_right}, "
                f"only {args.reference} right: {d.reference_only_right})"
            )
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    policy = GatePolicy(
        max_accuracy_drop=args.max_drop,
        max_invalid_rate=args.max_invalid,
        max_p95_latency_s=args.max_p95_s,
    )
    d = decide(
        _name(args.candidate),
        read_predictions(args.candidate),
        _name(args.baseline),
        read_predictions(args.baseline),
        policy,
    )
    print("PROMOTE" if d.promote else "REFUSE", f"{d.candidate} vs {d.baseline}")
    if d.delta:
        print(f"  accuracy {d.delta.delta:+.1%} [{d.delta.ci_low:+.1%}, {d.delta.ci_high:+.1%}]")
    for r in d.reasons:
        print(f"  - {r}")
    return 0 if d.promote else 1


def _print_stats(s: RunStats) -> None:
    def fmt(d: object) -> str:
        from sovereign.bench import Dist

        assert isinstance(d, Dist)
        flag = f" (unreliable: {','.join(d.unreliable)})" if d.unreliable else ""
        return f"p50={d.p50} p95={d.p95} p99={d.p99} n={d.n}{flag}"

    print(
        f"[{s.mode} load={s.load}] {s.requests} req, {s.errors} errors, {s.requests_per_s} req/s, "
        f"{s.output_tokens_per_s} out tok/s"
    )
    print(f"   TTFT {fmt(s.ttft_s)}")
    print(f"   TPOT {fmt(s.tpot_s)}")
    print(f"   e2e  {fmt(s.latency_s)}")


def _cmd_bench(args: argparse.Namespace) -> int:
    s = _splits()
    questions = s.test[: args.requests]
    prompts = (
        few_shot(questions, s.train[: args.shots])
        if args.workload == "few-shot"
        else zero_shot(questions)
    )
    extra = {"ignore_eos": True} if args.fixed_output else None
    client = OpenAICompatibleClient(args.base_url, args.model, extra_body=extra)

    async def go() -> list[RunStats]:
        out = []
        for c in [int(x) for x in args.concurrency.split(",")] if args.concurrency else []:
            out.append(
                await closed_loop(
                    client, prompts, concurrency=c, max_tokens=args.max_tokens, label=args.workload
                )
            )
        for r in [float(x) for x in args.rate.split(",")] if args.rate else []:
            out.append(
                await open_loop(
                    client, prompts, rate_per_s=r, max_tokens=args.max_tokens, label=args.workload
                )
            )
        await client.aclose()
        return out

    for stats in asyncio.run(go()):
        _print_stats(stats)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sovereign", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    default_model = "Qwen/Qwen3-4B-Instruct-2507"

    dl = sub.add_parser(
        "download", help="fetch the model and dataset on the machine that runs this"
    )
    dl.add_argument("names", nargs="*", help="assets to fetch (default: all; see --list)")
    dl.add_argument("--list", action="store_true", help="show assets, sizes and what needs them")
    dl.add_argument("--dry-run", action="store_true", help="show what would be fetched")
    dl.add_argument("--cache-dir", help="instead of the Hugging Face cache (HF_HOME)")
    dl.set_defaults(func=_cmd_download)

    sub.add_parser("data", help="PubMedQA splits and label balance").set_defaults(func=_cmd_data)

    m = sub.add_parser("memory", help="weight memory and KV-cache capacity per precision")
    m.add_argument("--model", default=default_model)
    m.add_argument("--config", help="a local config.json directory instead of the hub")
    m.add_argument("--gpu-gib", type=float, default=24.0)
    m.add_argument("--context", type=int, default=4096)
    m.set_defaults(func=_cmd_memory)

    sub.add_parser("smoke", help="CPU fine-tune/eval loop on a tiny model").set_defaults(
        func=_cmd_smoke
    )

    t = sub.add_parser("train", help="QLoRA / LoRA fine-tune on the PubMedQA train split")
    t.add_argument("--model", default=default_model)
    t.add_argument("--precision", choices=["nf4", "bf16"], default="nf4")
    t.add_argument("--output-dir", default="runs/qlora")
    t.add_argument("--epochs", type=float, default=3.0)
    t.add_argument("--r", type=int, default=16)
    t.add_argument("--alpha", type=int, default=32)
    t.set_defaults(func=_cmd_train)

    g = sub.add_parser("merge", help="fold a LoRA adapter into a bf16 copy of the base")
    g.add_argument("adapter_dir")
    g.add_argument("out_dir")
    g.add_argument("--model", default=default_model)
    g.set_defaults(func=_cmd_merge)

    z = sub.add_parser("compress", help="quantize the merged model for serving (llm-compressor)")
    z.add_argument("model_dir", help="the merged bf16 checkpoint (`sovereign merge`)")
    z.add_argument("out_dir")
    z.add_argument("--scheme", choices=["fp8", "w4a16"], default="fp8")
    z.add_argument("--calibration-samples", type=int, default=512, help="w4a16: train prompts")
    z.set_defaults(func=_cmd_compress)

    v = sub.add_parser("serve", help="print (or exec) the vLLM command")
    v.add_argument("--model", default=default_model)
    v.add_argument("--precision", choices=["bf16", "nf4", "fp8", "w4a16"], default="bf16")
    v.add_argument("--adapter", action="append", default=[], help="name=path, repeatable")
    v.add_argument("--port", type=int, default=8000)
    v.add_argument("--max-num-seqs", type=int, default=256)
    v.add_argument("--max-model-len", type=int, default=4096)
    v.add_argument("--no-prefix-caching", action="store_true")
    v.add_argument("--exec", action="store_true", help="replace this process with vllm serve")
    v.set_defaults(func=_cmd_serve)

    e = sub.add_parser("eval", help="answer the 500 test questions, write predictions")
    e.add_argument("--base-url")
    e.add_argument("--model")
    e.add_argument("--anthropic", action="store_true", help="the frontier reference")
    e.add_argument("--effort", default="high")
    e.add_argument("--out", required=True)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--concurrency", type=int, default=16)
    e.add_argument(
        "--constrained",
        action="store_true",
        help="decode the reply as exactly yes / no / maybe (vLLM structured outputs)",
    )
    e.set_defaults(func=_cmd_eval)

    c = sub.add_parser("compare", help="score prediction files; paired deltas vs a reference")
    c.add_argument("predictions", nargs="+")
    c.add_argument("--reference", help="system name (file stem) to compare against")
    c.set_defaults(func=_cmd_compare)

    gt = sub.add_parser("gate", help="promote or refuse a candidate against a baseline")
    gt.add_argument("--candidate", required=True)
    gt.add_argument("--baseline", required=True)
    gt.add_argument("--max-drop", type=float, default=0.02)
    gt.add_argument("--max-invalid", type=float, default=0.01)
    gt.add_argument("--max-p95-s", type=float, default=None)
    gt.set_defaults(func=_cmd_gate)

    b = sub.add_parser("bench", help="load an OpenAI-compatible server")
    b.add_argument("--base-url", required=True)
    b.add_argument("--model", required=True)
    b.add_argument("--workload", choices=["zero-shot", "few-shot"], default="zero-shot")
    b.add_argument("--shots", type=int, default=5)
    b.add_argument("--requests", type=int, default=200)
    b.add_argument("--concurrency", help="closed loop, e.g. 1,4,16,64")
    b.add_argument("--rate", help="open loop requests/s, e.g. 2,4,8")
    b.add_argument("--max-tokens", type=int, default=128)
    b.add_argument(
        "--fixed-output",
        action="store_true",
        help="ignore EOS so every request decodes exactly --max-tokens (vLLM)",
    )
    b.set_defaults(func=_cmd_bench)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ModuleNotFoundError as e:
        hint = _install_hint(e.name)
        if hint is None:
            raise
        print(f"error: {e.name} is not installed; run: pip install -r {hint}", file=sys.stderr)
        return 2
    except TypeError as e:  # the SDK found no credentials when the request was made
        if "Could not resolve authentication method" not in str(e):
            raise
        print("error: no Anthropic credentials; set ANTHROPIC_API_KEY (see .env.example)",
              file=sys.stderr)  # fmt: skip
        return 2


# optional module -> the requirements file that installs it
_OPTIONAL = {
    "torch": "requirements-train.txt",
    "transformers": "requirements-train.txt",
    "peft": "requirements-train.txt",
    "accelerate": "requirements-train.txt",
    "bitsandbytes": "requirements-train.txt",
    "llmcompressor": "requirements-compress.txt",
    "anthropic": "requirements-frontier.txt",
}


def _install_hint(module: str | None) -> str | None:
    name = module or ""
    return next((f for m, f in _OPTIONAL.items() if name == m or name.startswith(m + ".")), None)


if __name__ == "__main__":
    raise SystemExit(main())
