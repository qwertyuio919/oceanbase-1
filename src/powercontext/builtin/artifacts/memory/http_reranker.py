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

"""HTTP cross-encoder reranking with bounded batches and global ordering."""

from __future__ import annotations

import math
from typing import Any, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

HTTP_RERANK_POLICY_ID = "http-cross-encoder-v1"
DEFAULT_HTTP_RERANK_BATCH_SIZE = 32


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _RerankCandidate(_StrictModel):
    rank: int = Field(ge=1)
    text: str


class _RerankRequest(_StrictModel):
    query: str
    candidates: tuple[_RerankCandidate, ...]


class _RerankScore(_StrictModel):
    rank: int = Field(ge=1)
    score: float = Field(allow_inf_nan=False)


class _RerankResponse(_StrictModel):
    scores: tuple[_RerankScore, ...]


class HTTPMemoryReranker:
    """Score coarse candidates through one offline HTTP service."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = DEFAULT_HTTP_RERANK_BATCH_SIZE,
        policy_id: str = HTTP_RERANK_POLICY_ID,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch size must be a positive integer")
        if not endpoint.strip():
            raise ValueError("endpoint must not be blank")
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Send every global rank once and select from the full score pool."""

        _validate_bounds(len(candidates), limit)
        numbered = tuple(enumerate(candidates, start=1))
        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(numbered), self._batch_size):
            batch = numbered[start : start + self._batch_size]
            payload = _RerankRequest(
                query=query,
                candidates=tuple(_RerankCandidate(rank=rank, text=hit.text) for rank, hit in batch),
            )
            response = await self._client.post(self._endpoint, json=payload.model_dump(mode="json"))
            requests += 1
            response.raise_for_status()
            scores.update(_parse_scores(_response_payload(response), expected=tuple(rank for rank, _ in batch)))

        ordered = sorted(scores, key=lambda rank: (-scores[rank], rank))[:limit]
        return MemoryRerankDecision(
            selected_ranks=tuple(ordered),
            usage=InferenceUsage(requests=requests),
        )


def _response_payload(response: httpx.Response, /) -> Any:
    try:
        payload = response.json()
    except ValueError as error:
        raise ValueError("rerank response must be valid JSON") from error
    if not isinstance(payload, (dict, list)):
        raise ValueError("rerank response must be a JSON object")
    return payload


def _parse_scores(payload: Any, *, expected: tuple[int, ...]) -> dict[int, float]:
    if not isinstance(payload, dict) or set(payload) != {"scores"}:
        raise ValueError("rerank response must contain exactly the scores field")
    raw_scores = payload["scores"]
    if not isinstance(raw_scores, list):
        raise ValueError("rerank response scores must be a JSON array")
    wanted = set(expected)
    observed: dict[int, float] = {}
    for item in raw_scores:
        if not isinstance(item, dict) or set(item) != {"rank", "score"}:
            raise ValueError("each rerank score must contain exactly rank and score")
        rank = item["rank"]
        score = item["score"]
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError("rerank ranks must be integers")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("rerank scores must be numeric")
        if not math.isfinite(float(score)):
            raise ValueError("rerank scores must be finite")
        if rank not in wanted:
            raise ValueError("rerank scores must cover exactly the requested batch")
        if rank in observed:
            raise ValueError("rerank scores must reference each rank at most once")
        observed[rank] = float(score)
    if set(observed) != wanted:
        raise ValueError("rerank scores must cover exactly the requested batch")
    return observed


def _validate_bounds(candidate_count: int, limit: int) -> None:
    if candidate_count < 1:
        raise ValueError("rerank requires at least one candidate")
    if limit < 1 or limit > candidate_count:
        raise ValueError("rerank limit must be between one and the candidate count")


__all__ = ["DEFAULT_HTTP_RERANK_BATCH_SIZE", "HTTP_RERANK_POLICY_ID", "HTTPMemoryReranker"]
