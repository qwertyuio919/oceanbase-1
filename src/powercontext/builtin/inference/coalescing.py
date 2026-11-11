# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Coalesce concurrent Embedding work and cache successful vectors."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel

DEFAULT_MAX_BATCH_SIZE = 64
DEFAULT_COALESCE_MS = 5.0
DEFAULT_CACHE_SIZE = 1024


@dataclass
class _Entry:
    """One text waiting for (or already covered by) one underlying batch."""

    text: str
    future: asyncio.Future[EmbeddingVector]
    batch: _Batch | None = None


@dataclass
class _Batch:
    """One deduplicated request whose result fans out to every waiter."""

    texts: tuple[str, ...]
    entries: tuple[_Entry, ...]
    usage: InferenceUsage | None = None
    failed: bool = False


class CoalescingEmbeddingModel:
    """Share in-flight Embedding work across callers and cache successes."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        coalesce_ms: float = DEFAULT_COALESCE_MS,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        if not isinstance(max_batch_size, int) or isinstance(max_batch_size, bool) or max_batch_size < 1:
            raise ValueError("max batch size must be a positive integer")
        if coalesce_ms < 0:
            raise ValueError("coalesce window must not be negative")
        if not isinstance(cache_size, int) or isinstance(cache_size, bool) or cache_size < 0:
            raise ValueError("cache size must not be negative")
        self._model = model
        self.profile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_ms = float(coalesce_ms)
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._inflight: dict[str, _Entry] = {}
        self._pending: list[_Entry] = []
        self._flush_handle: asyncio.TimerHandle | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return vectors in call order while sharing provider work."""

        values = tuple(texts)
        if not values:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        entries: list[_Entry] = []
        vectors: dict[str, EmbeddingVector] = {}
        for text in _unique(values):
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                vectors[text] = cached
                continue
            entry = self._inflight.get(text)
            if entry is None:
                entry = _Entry(text=text, future=asyncio.get_running_loop().create_future())
                self._inflight[text] = entry
                self._pending.append(entry)
                self._schedule_flush()
            entries.append(entry)

        if entries:
            try:
                await asyncio.gather(*(asyncio.shield(entry.future) for entry in entries))
            except asyncio.CancelledError:
                raise
            except BaseException:
                # Waiters that share a failed batch all observe the same error, and the
                # entries stay out of the cache so a retry can issue a fresh request.
                raise
            for entry in entries:
                vectors[entry.text] = entry.future.result()

        usage = _aggregate_usage(entries)
        ordered = tuple(vectors[value] for value in values)
        return EmbeddingResult(vectors=ordered, usage=usage)

    def _schedule_flush(self) -> None:
        if self._flush_handle is not None:
            return
        loop = asyncio.get_running_loop()
        if self._coalesce_ms <= 0:
            self._flush_handle = loop.call_soon(self._flush)
        else:
            self._flush_handle = loop.call_later(self._coalesce_ms / 1_000, self._flush)

    def _flush(self) -> None:
        self._flush_handle = None
        if not self._pending:
            return
        chunk = self._pending[: self._max_batch_size]
        del self._pending[: len(chunk)]
        batch = _Batch(texts=tuple(entry.text for entry in chunk), entries=tuple(chunk))
        for entry in chunk:
            entry.batch = batch
        task = asyncio.create_task(self._run_batch(batch))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        if self._pending:
            self._schedule_flush()

    async def _run_batch(self, batch: _Batch, /) -> None:
        try:
            result = await self._model.embed(batch.texts)
        except asyncio.CancelledError:
            self._fail(batch, asyncio.CancelledError())
            raise
        except BaseException as error:
            self._fail(batch, error)
            return

        vectors = tuple(result.vectors)
        if len(vectors) != len(batch.texts):
            self._fail(batch, ValueError("provider must return one vector per batch text"))
            return
        batch.usage = result.usage
        for text, vector in zip(batch.texts, vectors, strict=True):
            entry = self._inflight.pop(text, None)
            self._remember(text, vector)
            if entry is not None and not entry.future.done():
                entry.future.set_result(vector)

    def _fail(self, batch: _Batch, error: BaseException, /) -> None:
        batch.failed = True
        for text in batch.texts:
            entry = self._inflight.pop(text, None)
            if entry is not None and not entry.future.done():
                entry.future.set_exception(error)

    def _remember(self, text: str, vector: EmbeddingVector, /) -> None:
        if self._cache_size == 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


def _unique(values: tuple[str, ...], /) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _aggregate_usage(entries: list[_Entry], /) -> InferenceUsage:
    """Charge each waited-on underlying batch once and keep provider counters."""

    batches: dict[int, _Batch] = {}
    for entry in entries:
        if entry.batch is not None:
            batches[id(entry.batch)] = entry.batch
    requests = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    for batch in batches.values():
        usage = batch.usage
        if usage is None:
            continue
        requests += usage.requests
        if usage.input_tokens is not None:
            input_tokens = (input_tokens or 0) + usage.input_tokens
        if usage.output_tokens is not None:
            output_tokens = (output_tokens or 0) + usage.output_tokens
    return InferenceUsage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens)


__all__ = ["CoalescingEmbeddingModel"]
