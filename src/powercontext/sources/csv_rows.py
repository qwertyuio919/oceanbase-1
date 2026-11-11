from __future__ import annotations

import csv
import io
import urllib.parse
from typing import Tuple, List, Any
from pydantic import BaseModel, ConfigDict, Field

from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.adapters import SourceAdapter

# ==========================================
# 1. 定义输入和输出的数据模型 (Pydantic)
# ==========================================

class CSVDocumentInput(BaseModel):
    document_id: str
    primary_key: str
    content: bytes

class CSVRowInput(BaseModel):
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: Tuple[str, ...]
    values: Tuple[str, ...]

# ==========================================
# 2. 严格继承官方要求的 Source 基类
# ==========================================

class CSVRowSource(Source):
    # 根据 models.py 中的定义，必须包含这些字段
    name: str = "csv-row"
    definition_version: str = "1"
    materialization: SourceMaterialization = SourceMaterialization.CAPTURED
    
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: Tuple[str, ...]
    values: Tuple[str, ...]

# ==========================================
# 3. 实现解析逻辑和 ID 生成逻辑
# ==========================================

def parse_csv_document(value: CSVDocumentInput) -> Tuple[CSVRowInput, ...]:
    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError(f"Invalid UTF-8 encoding: {e}")

    reader = csv.reader(io.StringIO(text), strict=True)
    
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("Empty file")

    if not header or all(not col.strip() for col in header):
        raise ValueError("Invalid header")
    if len(header) != len(set(header)):
        raise ValueError("Duplicate columns")

    if value.primary_key not in header:
        raise ValueError(f"Primary key '{value.primary_key}' not found")

    pk_index = header.index(value.primary_key)
    seen_pks = set()
    results = []

    for row_num, row in enumerate(reader, start=2):
        if not row: continue
        if len(row) != len(header):
            raise ValueError(f"Row {row_num} length mismatch")

        pk_value = row[pk_index]
        if not pk_value.strip():
            raise ValueError(f"Row {row_num}: empty primary key")
        if pk_value in seen_pks:
            raise ValueError(f"Row {row_num}: duplicate primary key")
        seen_pks.add(pk_value)

        results.append(
            CSVRowInput(
                document_id=value.document_id,
                primary_key=value.primary_key,
                primary_key_value=pk_value,
                columns=tuple(header),
                values=tuple(row)
            )
        )
    return tuple(results)

def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    encoded_doc = urllib.parse.quote(document_id, safe="")
    encoded_pk = urllib.parse.quote(primary_key_value, safe="")
    return f"csv:{encoded_doc}:{encoded_pk}"

# ==========================================
# 4. 实现官方要求的适配器 (Adapter)
# ==========================================

class CSVRowAdapter:
    input_class = CSVDocumentInput
    source_class = CSVRowSource
    name = "csv-row"

    # 必须实现异步方法 resolve
    async def resolve(self, value: CSVDocumentInput) -> List[CSVRowSource]:
        rows = parse_csv_document(value)
        sources = []
        for row in rows:
            sid = csv_row_source_id(row.document_id, row.primary_key_value)
            source = CSVRowSource(
                name=self.name,
                definition_version="1",
                materialization=SourceMaterialization.CAPTURED,
                document_id=row.document_id,
                primary_key=row.primary_key,
                primary_key_value=row.primary_key_value,
                columns=row.columns,
                values=row.values
            )
            # 如果 Source 基类有 source_id 属性，在这里赋值
            if hasattr(source, 'source_id'):
                source.source_id = sid
            sources.append(source)
        return sources

    # 必须实现异步方法 read
    async def read(self, source: CSVRowSource) -> str:
        content_dict = dict(zip(source.columns, source.values))
        import json
        return json.dumps(content_dict, ensure_ascii=False, separators=(',', ':'))

# ==========================================
# 5. 导出 Definition 供系统注册
# ==========================================

# 很多框架只需要一个包含 input_class, source_class, adapter 实例的字典或对象
CSV_ROW_SOURCE_DEFINITION = {
    "name": "csv-row",
    "version": "1",
    "input_class": CSVDocumentInput,
    "source_class": CSVRowSource,
    "adapter": CSVRowAdapter()
}
