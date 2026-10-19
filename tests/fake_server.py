"""A real HTTP server that speaks just enough of the OpenAI streaming chat API, with known timing.

Not a model: each request waits for one of `slots` (a crude stand-in for the server's batch
capacity), sleeps `prefill_s`, then streams `tokens` chunks `itl_s` apart, a usage chunk and
`[DONE]`. Because the timing is known, the benchmark can be tested as a measuring instrument: TTFT
should come back as about `prefill_s` plus any queueing, TPOT as about `itl_s`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field


@dataclass
class FakeServer:
    tokens: int = 5
    prefill_s: float = 0.02
    itl_s: float = 0.005
    slots: int = 1000
    reply: str = "yes"
    status: int = 200
    cached_tokens: int | None = None  # report usage.prompt_tokens_details.cached_tokens
    fail_mid_stream: bool = False
    requests: list[dict[str, object]] = field(default_factory=list)
    _sem: asyncio.Semaphore | None = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        length = 0
        for line in head.decode().split("\r\n"):
            if line.lower().startswith("content-length:"):
                length = int(line.split(":", 1)[1])
        body = json.loads(await reader.readexactly(length)) if length else {}
        self.requests.append(body)
        if self.status != 200:
            writer.write(
                f"HTTP/1.1 {self.status} Error\r\nContent-Length: 4\r\n"
                "Connection: close\r\n\r\noops".encode()
            )
            await writer.drain()
            writer.close()
            return
        assert self._sem is not None
        async with self._sem:
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Cache-Control: no-cache\r\nConnection: close\r\n\r\n"
            )
            role = {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}
            writer.write(f"data: {json.dumps(role)}\n\n".encode())  # vLLM's first chunk
            await asyncio.sleep(self.prefill_s)
            if self.fail_mid_stream:
                err = {"error": {"message": "engine died", "code": 500}}
                writer.write(f"data: {json.dumps(err)}\n\ndata: [DONE]\n\n".encode())
                await writer.drain()
                writer.close()
                return
            pieces = [self.reply] + ["."] * (self.tokens - 1)
            for i, piece in enumerate(pieces):
                if i:
                    await asyncio.sleep(self.itl_s)
                chunk = {
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]
                }
                writer.write(f"data: {json.dumps(chunk)}\n\n".encode())
                await writer.drain()
            end = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            usage: dict[str, object] = {"prompt_tokens": 42, "completion_tokens": self.tokens}
            if self.cached_tokens is not None:
                usage["prompt_tokens_details"] = {"cached_tokens": self.cached_tokens}
            final = {"choices": [], "usage": usage}
            writer.write(f": keep-alive\n\ndata: {json.dumps(end)}\n\n".encode())
            writer.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())
            await writer.drain()
        writer.close()

    @asynccontextmanager
    async def running(self) -> AsyncIterator[str]:
        self._sem = asyncio.Semaphore(self.slots)
        server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            yield f"http://127.0.0.1:{port}/v1"
        finally:
            server.close()
            await server.wait_closed()
