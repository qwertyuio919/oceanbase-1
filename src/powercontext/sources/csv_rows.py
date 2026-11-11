import csv
import io
import urllib.parse
from typing import Tuple, Any
from pydantic import BaseModel

# --- 数据类定义 ---
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

# 假设 Base Source 类如此定义，具体视原项目情况而定
class Source:
    def __init__(self, source_id: str):
        self.source_id = source_id

class CSVRowSource(Source):
    def __init__(self, source_id: str, document_id: str, primary_key: str, primary_key_value: str, columns: Tuple[str, ...], values: Tuple[str, ...]):
        super().__init__(source_id)
        self.document_id = document_id
        self.primary_key = primary_key
        self.primary_key_value = primary_key_value
        self.columns = columns
        self.values = values

# --- 你需要实现的函数 ---

def parse_csv_document(value: CSVDocumentInput) -> Tuple[CSVRowInput, ...]:
    try:
        # 按照题目要求：严格 UTF-8 解码，接受 BOM
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError(f"Invalid UTF-8 encoding: {e}")

    reader = csv.reader(io.StringIO(text), strict=True)
    
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("Empty file")

    # 校验表头：非空、非纯空白、唯一
    if not header or all(not col.strip() for col in header):
        raise ValueError("Invalid header: empty or all whitespace")
    if len(header) != len(set(header)):
        raise ValueError("Invalid header: duplicate columns")

    if value.primary_key not in header:
        raise ValueError(f"Primary key '{value.primary_key}' not found in header")

    pk_index = header.index(value.primary_key)
    seen_pks = set()
    results = []

    for row_num, row in enumerate(reader, start=2):
        if not row: # 忽略完全空行
            continue
            
        if len(row) != len(header):
            raise ValueError(f"Row {row_num}: expected {len(header)} fields, got {len(row)}")

        pk_value = row[pk_index]
        # 主键不能为空或纯空白
        if not pk_value.strip():
            raise ValueError(f"Row {row_num}: primary key value is empty or whitespace")
        # 主键必须唯一
        if pk_value in seen_pks:
            raise ValueError(f"Row {row_num}: duplicate primary key value '{pk_value}'")
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
    # 按题目要求进行百分号编码，safe="" 表示连斜杠等字符也要编码
    encoded_doc = urllib.parse.quote(document_id, safe="")
    encoded_pk = urllib.parse.quote(primary_key_value, safe="")
    return f"csv:{encoded_doc}:{encoded_pk}"
