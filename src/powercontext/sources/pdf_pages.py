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

"""Page-level PDF Sources: parse every physical page, then recall exact pages.

One Source is kept per physical page, so every citation traces back to the whole
document digest plus the physical page number. Retrieval scores the pages held in
the Memory offline with BM25 and asks the Memory retrieval engine for extra pages
only when the lexical ranking cannot fill the requested limit.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from urllib.parse import quote

from pydantic import BaseModel
from pypdf import PdfReader

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps sources free of runtime builtin imports
    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.builtin.runtime import ScopedMemoryApplication

PDF_PAGE_SOURCE_NAME = "pdf-page"
PDF_PAGE_SOURCE_VERSION = "1"

_TOKEN_PATTERN = re.compile(r"[a-z]+|\d+")
_BM25_K1 = 1.2
_BM25_B = 0.75
_CACHE_LIMIT = 4


class PDFDocumentInput(BaseModel):
    """One complete original PDF addressed by a stable document id."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One physical page resolved against the whole document digest."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """Captured page whose identity is the document digest plus the physical page."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def parse_pdf_document(value: PDFDocumentInput, /) -> tuple[PDFPageInput, ...]:
    """Parse every physical page of one PDF and fail atomically on any error."""

    document_id = _validated_document_id(value.document_id)
    content = value.content
    if not content:
        raise ValueError("PDF document is empty")
    digest = hashlib.sha256(content).hexdigest()

    reader = _open_reader(content)
    page_count = _page_count(reader)
    pages: list[PDFPageInput] = []
    for index in range(page_count):
        try:
            text = reader.pages[index].extract_text()
        except Exception as error:  # noqa: BLE001 - any reader failure must fail the whole parse
            raise ValueError(f"PDF page {index + 1} could not be parsed") from error
        pages.append(
            PDFPageInput(
                document_id=document_id,
                document_sha256=digest,
                page_number=index + 1,
                text="" if text is None else str(text),
            )
        )
    return tuple(pages)


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int, /) -> str:
    """Build the stable page identity from document id, digest and physical page."""

    encoded_document = quote(_validated_document_id(document_id), safe="")
    _validated_digest(document_sha256)
    if page_number < 1:
        raise ValueError("page number must start at 1")
    identity = f"pdf:{encoded_document}:{document_sha256}:{page_number}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        identity = f"pdf:{hashlib.sha256(document_id.encode()).hexdigest()[:32]}:{document_sha256}:{page_number}"
        if len(identity) > MAX_SOURCE_ID_LENGTH:
            raise ValueError(f"page identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return identity


async def search_pdf_pages(
    memory: ScopedMemoryApplication,
    question: str,
    limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Return at most ``limit`` exact citations to distinct pages for one question."""

    if limit <= 0:
        return ()
    normalized = " ".join(question.split())
    if not normalized:
        return ()

    index = await _page_index(memory)
    if not index.entries:
        return ()

    selected: list[MemoryCitation] = []
    seen: set[str] = set()
    for entry_index in index.rank(normalized)[:limit]:
        page = index.entries[entry_index]
        if page.source_id in seen:
            continue
        seen.add(page.source_id)
        selected.append(page.citation)
    if len(selected) >= limit:
        return tuple(selected)

    for page in await _engine_fallback(memory, normalized, limit):
        if page.source_id in seen:
            continue
        seen.add(page.source_id)
        selected.append(page.citation)
        if len(selected) >= limit:
            break
    return tuple(selected)


@dataclass(frozen=True)
class _IndexedPage:
    citation: MemoryCitation
    source_id: str
    text: str


class _PageIndex:
    """Cached BM25 statistics over every page currently held in one Memory."""

    def __init__(self, entries: list[_IndexedPage]) -> None:
        self.entries = entries
        token_counts = [Counter(_tokenize(entry.text)) for entry in entries]
        lengths = [max(sum(tokens.values()), 1) for tokens in token_counts]
        average_length = sum(lengths) / len(entries)
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for entry_index, tokens in enumerate(token_counts):
            for term, frequency in tokens.items():
                self._postings.setdefault(term, []).append((entry_index, frequency))
        self._document_frequencies = {term: len(postings) for term, postings in self._postings.items()}
        self._length_factors = [
            _BM25_K1 * (1.0 - _BM25_B + _BM25_B * length / average_length) for length in lengths
        ]

    def rank(self, question: str) -> list[int]:
        """Rank pages by BM25 relevance, breaking ties by document order."""

        scores = [0.0] * len(self.entries)
        document_total = len(self.entries)
        for term in dict.fromkeys(_tokenize(question)):
            postings = self._postings.get(term)
            if not postings:
                continue
            document_frequency = self._document_frequencies[term]
            idf = math.log(1.0 + (document_total - document_frequency + 0.5) / (document_frequency + 0.5))
            for entry_index, frequency in postings:
                scores[entry_index] += (
                    idf * frequency * (_BM25_K1 + 1.0) / (frequency + self._length_factors[entry_index])
                )
        return sorted(range(len(self.entries)), key=lambda entry_index: (-scores[entry_index], entry_index))


_PAGE_INDEX_CACHE: dict[tuple[str, str], _PageIndex] = {}


async def _page_index(memory: ScopedMemoryApplication) -> _PageIndex:
    page = await memory.list()
    if page.memory_ref is None:
        return _PageIndex([])
    cache_key = (getattr(memory, "scope_id", ""), str(page.memory_ref))
    cached = _PAGE_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    entries: list[_IndexedPage] = []
    for record in page.entries:
        sources = record.entry.sources
        if not sources or not record.entry.text.strip():
            continue
        entries.append(_IndexedPage(citation=record.citation, source_id=sources[0].source_id, text=record.entry.text))
    index = _PageIndex(entries) if entries else _PageIndex([])
    if len(_PAGE_INDEX_CACHE) >= _CACHE_LIMIT:
        _PAGE_INDEX_CACHE.pop(next(iter(_PAGE_INDEX_CACHE)))
    _PAGE_INDEX_CACHE[cache_key] = index
    return index


async def _engine_fallback(memory: ScopedMemoryApplication, question: str, limit: int) -> list[_IndexedPage]:
    """Ask the Memory retrieval engine for pages the lexical ranking did not reach."""

    from powercontext.builtin.artifacts.memory.models import MemoryCitation  # noqa: PLC0415
    from powercontext.builtin.runtime import GetMemoryEntryRequest, SearchMemoryRequest  # noqa: PLC0415

    try:
        result = await memory.search(SearchMemoryRequest(query=question, mode="fts", limit=min(50, max(limit, 20))))
    except Exception:  # noqa: BLE001 - retrieval stays best-effort inside the offline ranker
        return []

    pages: list[_IndexedPage] = []
    for hit in result.hits:
        citation = MemoryCitation(
            memory_ref=hit.memory_ref,
            entry_id=hit.entry_id,
            entry_version_id=hit.entry_version_id,
        )
        try:
            entry = await memory.get(GetMemoryEntryRequest(citation=citation))
        except Exception:  # noqa: BLE001 - a stale hit is simply skipped
            continue
        sources = entry.entry.sources
        if not sources:
            continue
        pages.append(_IndexedPage(citation=citation, source_id=sources[0].source_id, text=entry.entry.text))
    return pages


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _open_reader(content: bytes) -> PdfReader:
    try:
        reader = PdfReader(io.BytesIO(content), strict=False)
    except Exception as error:  # noqa: BLE001 - any reader failure means the document is invalid
        raise ValueError("PDF document is not readable") from error
    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as error:  # noqa: BLE001 - an unusable permission marker is invalid input
            raise ValueError("PDF document requires a password") from error
        if not unlocked:
            raise ValueError("PDF document requires a password")
    return reader


def _page_count(reader: PdfReader) -> int:
    try:
        count = len(reader.pages)
    except Exception as error:  # noqa: BLE001 - a broken page tree invalidates the document
        raise ValueError("PDF document pages are not readable") from error
    if count < 1:
        raise ValueError("PDF document has no pages")
    return count


def _validated_document_id(document_id: str) -> str:
    if not document_id.strip():
        raise ValueError("document id must not be blank")
    if document_id != document_id.strip():
        raise ValueError("document id must not contain surrounding whitespace")
    return document_id


def _validated_digest(document_sha256: str) -> str:
    if len(document_sha256) != 64 or any(character not in "0123456789abcdef" for character in document_sha256):
        raise ValueError("document digest must be a lowercase hexadecimal SHA-256")
    return document_sha256


class PDFPageSourceAdapter:
    """Resolve and read page Sources without re-parsing the document."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        identity = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=identity,
            source_id=identity,
            definition_version=PDF_PAGE_SOURCE_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose one physical page as canonical textual evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION: SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = cast(
    "SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput]",
    AdapterSourceDefinition(
        adapter=PDFPageSourceAdapter(),
        version=PDF_PAGE_SOURCE_VERSION,
        projections=(PDFPageTextEvidenceProjection(),),
    ),
)


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDF_PAGE_SOURCE_VERSION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "PDFPageSourceAdapter",
    "PDFPageTextEvidenceProjection",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
