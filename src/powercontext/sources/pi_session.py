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

"""Capture the raw visible events of one selected Pi session branch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import ConnectorRunCompletion, ConnectorRunSession, ConnectorRunStatus
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
PI_HISTORY_VERSION = "1"

_CONTROL_TYPES = frozenset({"session"})
_TEXT_BLOCK_TYPES = frozenset({"text"})

# A single Memory entry is bounded far below the size of real tool output, so the
# Source keeps the full text and only the projection consumer splits it.
_CHECKPOINT_VERSION = 1


class PiSessionInput(BaseModel):
    """One complete Pi session archive plus the branch leaf to capture."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible Pi event on the selected branch, with its original record."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """Captured Pi event whose identity is the session id plus the event id."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def parse_pi_session(value: PiSessionInput, /) -> tuple[PiEventInput, ...]:
    """Return every visible event on the selected leaf's ancestor chain."""

    records = _read_records(value.content)
    header, nodes = _validate_graph(records)
    leaf_id = value.leaf_id if value.leaf_id is not None else nodes[-1]["id"]
    if leaf_id not in {node["id"] for node in nodes}:
        raise ValueError("leaf id must name one event in the session")

    session_id = _header_session_id(header)
    by_id = {node["id"]: node for node in nodes}
    chain: list[dict[str, Any]] = []
    current: str | None = leaf_id
    while current is not None:
        node = by_id[current]
        chain.append(node)
        current = node.get("parentId")
    chain.reverse()

    events: list[PiEventInput] = []
    for node in chain:
        visible = _visible_event(session_id, node)
        if visible is not None:
            events.append(visible)
    return tuple(events)


def pi_event_source_id(session_id: str, event_id: str, /) -> str:
    """Build the stable event identity from its two percent-encoded components."""

    if not session_id.strip() or not event_id.strip():
        raise ValueError("session and event ids must not be blank")
    identity = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"event identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return identity


def pi_record_digest(record: dict[str, JsonValue], /) -> str:
    """Fingerprint one original record so rewritten history stays detectable."""

    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PiEventSourceAdapter:
    """Resolve and read Pi event Sources without touching session files."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        identity = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=identity,
            source_id=identity,
            definition_version=PI_HISTORY_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose one event as canonical textual evidence with its identity metadata."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION: SourceDefinition[PiEventInput, PiEventSource, PiEventInput] = cast(
    "SourceDefinition[PiEventInput, PiEventSource, PiEventInput]",
    AdapterSourceDefinition(
        adapter=PiEventSourceAdapter(),
        version=PI_HISTORY_VERSION,
        projections=(PiEventTextEvidenceProjection(),),
    ),
)


class PiSessionConnector:
    """Import only the still-unimported visible events of one Pi session branch."""

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_HISTORY_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: Path | str, leaf_id: str | None = None, /) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        events = parse_pi_session(PiSessionInput(content=self._path.read_bytes(), leaf_id=self._leaf_id))
        imported = _checkpoint_events(session.checkpoint)
        digests = {pi_event_source_id(event.session_id, event.event_id): pi_record_digest(event.record) for event in events}

        # Validate every already-imported event before submitting anything new so a
        # rewritten history cannot leave the checkpoint ahead of the stored evidence.
        for identity, digest in digests.items():
            known = imported.get(identity)
            if known is not None and known != digest:
                raise ValueError("imported Pi events were rewritten in the session file")

        proposed = dict(imported)
        for event in events:
            identity = pi_event_source_id(event.session_id, event.event_id)
            if identity in imported:
                continue
            await session.submit(identity, PI_EVENT_SOURCE_NAME, event)
            proposed[identity] = digests[identity]

        checkpoint: JsonValue = {"version": _CHECKPOINT_VERSION, "events": proposed}
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint)


def _checkpoint_events(checkpoint: JsonValue | None, /) -> dict[str, str]:
    if checkpoint is None:
        return {}
    if not isinstance(checkpoint, dict):
        raise ValueError("Pi session checkpoint must be a JSON object")
    events = checkpoint.get("events", {})
    if not isinstance(events, dict):
        raise ValueError("Pi session checkpoint events must be a JSON object")
    return {str(key): str(value) for key, value in events.items()}


def _read_records(content: bytes, /) -> list[dict[str, Any]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session archive is not valid UTF-8") from error

    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("Pi session archive contains malformed JSON") from error
        if not isinstance(record, dict):
            raise ValueError("every Pi session record must be a JSON object")
        records.append(record)
    if not records:
        raise ValueError("Pi session archive is empty")
    return records


def _header_session_id(header: dict[str, Any], /) -> str:
    if header.get("type") != "session":
        raise ValueError("the first Pi session record must be the session header")
    if header.get("version") != 3:
        raise ValueError("the Pi session header must declare version 3")
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("the Pi session header must declare a non-empty id")
    return session_id


def _validate_graph(records: list[dict[str, Any],], /) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header = records[0]
    nodes = records[1:]
    _header_session_id(header)
    seen: set[str] = set()
    for node in nodes:
        event_id = node.get("id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("every Pi session record must declare a non-empty id")
        if event_id in seen:
            raise ValueError("Pi session record ids must be unique")
        parent_id = node.get("parentId")
        if parent_id is not None:
            if not isinstance(parent_id, str) or parent_id not in seen:
                raise ValueError("every Pi session parent must already have appeared")
        seen.add(event_id)
    if not nodes:
        raise ValueError("Pi session archive has no branch nodes")
    return header, nodes  # type: ignore[return-value]


def _visible_event(session_id: str, node: dict[str, Any], /) -> PiEventInput | None:
    event_type = node.get("type")
    if not isinstance(event_type, str):
        raise ValueError("every Pi session record must declare a type")

    role: str | None = None
    body = ""
    if event_type == "message":
        message = node.get("message")
        if not isinstance(message, dict):
            raise ValueError("Pi message events must declare a message object")
        message_role = message.get("role")
        if not isinstance(message_role, str) or not message_role.strip():
            raise ValueError("Pi message events must declare a role")
        role = message_role
        body = _message_body(message_role, message)
    elif event_type in {"compaction", "branch_summary"}:
        role = event_type
        summary = node.get("summary")
        body = summary if isinstance(summary, str) else ""
    elif event_type == "custom_message":
        role = "custom"
        body = _block_text(node.get("content"))
    else:
        return None

    if not body.strip():
        return None
    return PiEventInput(
        session_id=session_id,
        event_id=node["id"],
        parent_id=node.get("parentId"),
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=node,
    )


def _message_body(role: str, message: dict[str, Any], /) -> str:
    if role == "toolResult":
        name = message.get("toolName")
        call_id = message.get("toolCallId")
        text = _block_text(message.get("content"))
        return f"[tool_result {name} {call_id}] {text}"
    if role == "bashExecution":
        command = message.get("command")
        output = message.get("output")
        return f"$ {command}\n{output}"
    return _block_text(message.get("content"))


def _block_text(content: object, /) -> str:
    """Keep visible text blocks and tool calls; drop thinking and images."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in _TEXT_BLOCK_TYPES:
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif block_type == "toolCall":
            arguments = block.get("arguments")
            payload = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            parts.append(f"[tool_call {block.get('name')} {block.get('id')}] {payload}")
    return "\n".join(part for part in parts if part.strip())


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_SESSION_CONNECTOR_NAME",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
    "pi_record_digest",
]
