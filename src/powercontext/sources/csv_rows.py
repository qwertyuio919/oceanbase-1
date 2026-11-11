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

"""Row-level CSV Sources that keep one retrievable record per table row."""

from __future__ import annotations

import csv
import io
import json
from typing import cast
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_NAME = "csv-row"
CSV_ROW_SOURCE_VERSION = "1"


class CSVDocumentInput(BaseModel):
    """One complete offline CSV document addressed by a stable document id."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One physical CSV data row resolved against its header."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """Captured dataset row whose identity is the document id plus primary key."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def parse_csv_document(value: CSVDocumentInput, /) -> tuple[CSVRowInput, ...]:
    """Parse and fully validate one CSV document before any row is returned."""

    document_id = _validated_document_id(value.document_id)
    primary_key = _validated_primary_key(value.primary_key)
    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV document is not valid UTF-8") from error

    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("CSV document is empty") from None
    _validate_header(header)
    if primary_key not in header:
        raise ValueError("primary key must match one header column exactly")
    key_index = header.index(primary_key)

    rows: list[CSVRowInput] = []
    seen: set[str] = set()
    try:
        for record in reader:
            if not record:
                continue
            if len(record) != len(header):
                raise ValueError("CSV record width must equal the header width")
            primary_key_value = record[key_index]
            if not primary_key_value.strip():
                raise ValueError("primary key value must not be blank")
            if primary_key_value in seen:
                raise ValueError("primary key values must be unique inside one document")
            seen.add(primary_key_value)
            rows.append(
                CSVRowInput(
                    document_id=document_id,
                    primary_key=primary_key,
                    primary_key_value=primary_key_value,
                    columns=tuple(header),
                    values=tuple(record),
                )
            )
    except csv.Error as error:
        raise ValueError("CSV document is malformed") from error
    return tuple(rows)


def csv_row_source_id(document_id: str, primary_key_value: str, /) -> str:
    """Build the stable row identity from its two percent-encoded components."""

    encoded_document = quote(_validated_document_id(document_id), safe="")
    if not primary_key_value.strip():
        raise ValueError("primary key value must not be blank")
    encoded_key = quote(primary_key_value, safe="")
    identity = f"csv:{encoded_document}:{encoded_key}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"row identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return identity


class CSVRowSourceAdapter:
    """Resolve and read row Sources without re-parsing the document."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        identity = csv_row_source_id(value.document_id, value.primary_key_value)
        if len(value.columns) != len(value.values):
            raise ValueError("row columns and values must describe the same fields")
        return CSVRowSource(
            name=identity,
            source_id=identity,
            definition_version=CSV_ROW_SOURCE_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose one row as canonical textual evidence keyed by its header order."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_DEFINITION: SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = cast(
    "SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput]",
    AdapterSourceDefinition(
        adapter=CSVRowSourceAdapter(),
        version=CSV_ROW_SOURCE_VERSION,
        projections=(CSVRowTextEvidenceProjection(),),
    ),
)


def _validated_document_id(document_id: str) -> str:
    if not document_id.strip():
        raise ValueError("document id must not be blank")
    if document_id != document_id.strip():
        raise ValueError("document id must not contain surrounding whitespace")
    return document_id


def _validated_primary_key(primary_key: str) -> str:
    if not primary_key.strip():
        raise ValueError("primary key must not be blank")
    return primary_key


def _validate_header(header: list[str]) -> None:
    if not header:
        raise ValueError("CSV header must not be empty")
    for column in header:
        if not column.strip():
            raise ValueError("CSV header columns must not be blank")
    if len(set(header)) != len(header):
        raise ValueError("CSV header columns must be unique")


__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSV_ROW_SOURCE_VERSION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "CSVRowSourceAdapter",
    "CSVRowTextEvidenceProjection",
    "csv_row_source_id",
    "parse_csv_document",
]
