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

"""Prefix-reusing Context Packs that keep prepared context transfer incremental."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from typing import Any

PACK_SCHEMA = "powercontext.context-pack.v1"
DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES = 512

_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_UNSIGNED_PACK_FIELDS = frozenset({"schema", "items"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})


class ContextPackError(ValueError):
    """Raised when a Context Pack or Delta violates its wire contract."""


@dataclass(frozen=True, slots=True)
class ContextPack:
    """One complete prepared context split into deterministic bounded chunks."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...], /) -> ContextPack:
        payload_items = tuple(bytes(item) for item in items)
        digest = _pack_digest(payload_items)
        data, byte_count = _rendered_pack(payload_items, digest)
        return cls(items=payload_items, digest=digest, byte_count=byte_count)

    @classmethod
    def from_content(cls, content: str | None, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes < 1:
            raise ContextPackError("chunk size must be a positive integer")
        return cls.build(_chunk_content(content or "", chunk_bytes))

    @classmethod
    def from_bytes(cls, data: bytes, /) -> ContextPack:
        payload = _decode_payload(data, _PACK_FIELDS, "context pack")
        if payload.get("schema") != PACK_SCHEMA:
            raise ContextPackError("context pack schema is not recognized")
        items = _decode_items(payload.get("items"))
        digest = _pack_digest(items)
        if payload.get("digest") != digest:
            raise ContextPackError("context pack digest does not match its items")
        byte_count = payload.get("byte_count")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count != len(data):
            raise ContextPackError("context pack byte count does not match its encoding")
        return cls(items=items, digest=digest, byte_count=byte_count)

    def to_bytes(self, /) -> bytes:
        data = _rendered_pack(self.items, self.digest)[0]
        if len(data) != self.byte_count:
            raise ContextPackError("context pack byte count does not match its encoding")
        return data


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """One transport delta carrying the target suffix after a sharable prefix."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack, /) -> ContextPackDelta:
        prefix_items = _common_prefix_length(base.items, target.items)
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix_items,
            appended=target.items[prefix_items:],
        )

    @classmethod
    def from_bytes(cls, data: bytes, /) -> ContextPackDelta:
        payload = _decode_payload(data, _DELTA_FIELDS, "context pack delta")
        if payload.get("schema") != DELTA_SCHEMA:
            raise ContextPackError("context pack delta schema is not recognized")
        base_digest = payload.get("base_digest")
        target_digest = payload.get("target_digest")
        if not isinstance(base_digest, str) or not isinstance(target_digest, str):
            raise ContextPackError("context pack delta digests must be strings")
        prefix_items = payload.get("prefix_items")
        if not isinstance(prefix_items, int) or isinstance(prefix_items, bool) or prefix_items < 0:
            raise ContextPackError("context pack delta prefix must be a non-negative integer")
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix_items,
            appended=_decode_items(payload.get("appended")),
        )

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self, /) -> bytes:
        return _encode(
            {
                "schema": DELTA_SCHEMA,
                "base_digest": self.base_digest,
                "target_digest": self.target_digest,
                "prefix_items": self.prefix_items,
                "appended": [base64.b64encode(item).decode("ascii") for item in self.appended],
            }
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta, /) -> ContextPack:
    """Rebuild the target pack from one base pack and one received delta."""

    if delta.base_digest != base.digest:
        raise ContextPackError("context pack delta does not apply to this base pack")
    if delta.prefix_items > len(base.items):
        raise ContextPackError("context pack delta prefix exceeds the base chunk count")
    target = ContextPack.build((*base.items[: delta.prefix_items], *delta.appended))
    if target.digest != delta.target_digest:
        raise ContextPackError("context pack delta target digest does not match the rebuilt pack")
    return target


def _chunk_content(content: str, chunk_bytes: int) -> tuple[bytes, ...]:
    data = content.encode("utf-8")
    chunks: list[bytes] = []
    start = 0
    while start < len(data):
        end = min(start + chunk_bytes, len(data))
        while end > start:
            try:
                data[start:end].decode("utf-8")
            except UnicodeDecodeError:
                end -= 1
                continue
            break
        if end == start:
            raise ContextPackError("chunk size cannot hold one UTF-8 character")
        chunks.append(data[start:end])
        start = end
    return tuple(chunks)


def _common_prefix_length(base: tuple[bytes, ...], target: tuple[bytes, ...], /) -> int:
    shared = 0
    while shared < len(base) and shared < len(target) and base[shared] == target[shared]:
        shared += 1
    return shared


def _pack_digest(items: tuple[bytes, ...], /) -> str:
    unsigned = {
        "schema": PACK_SCHEMA,
        "items": [base64.b64encode(item).decode("ascii") for item in items],
    }
    return "sha256:" + hashlib.sha256(_encode(unsigned)).hexdigest()


def _rendered_pack(items: tuple[bytes, ...], digest: str, /) -> tuple[bytes, int]:
    """Resolve ``byte_count`` against its own encoded length."""

    byte_count = 0
    for _ in range(16):
        data = _encode(
            {
                "schema": PACK_SCHEMA,
                "items": [base64.b64encode(item).decode("ascii") for item in items],
                "digest": digest,
                "byte_count": byte_count,
            }
        )
        if len(data) == byte_count:
            return data, byte_count
        byte_count = len(data)
    raise ContextPackError("context pack byte count did not converge")


def _encode(payload: dict[str, Any], /) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _decode_payload(data: bytes, fields: frozenset[str], label: str, /) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextPackError(f"{label} is not canonical UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ContextPackError(f"{label} must be a JSON object")
    if set(payload) != fields:
        raise ContextPackError(f"{label} must contain exactly {sorted(fields)}")
    return payload


def _decode_items(value: object, /) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError("chunk list must be a JSON array")
    items: list[bytes] = []
    for entry in value:
        if not isinstance(entry, str):
            raise ContextPackError("chunks must be Base64 strings")
        try:
            items.append(base64.b64decode(entry.encode("ascii"), validate=True))
        except (binascii.Error, UnicodeEncodeError) as error:
            raise ContextPackError("chunk is not valid Base64") from error
    return tuple(items)


__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "DELTA_SCHEMA",
    "PACK_SCHEMA",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
