"""CMoney QueryTable / QueryTableLight API helper.

Place this file at: table_rag_multi/table_query.py
"""
from __future__ import annotations

import csv
import io
import json
import os
from typing import Any, Optional

import requests


def _get_base_url() -> str:
    """Return Adoxc service base URL; override with CM_ADOXC_BASE."""
    return os.getenv(
        "CM_ADOXC_BASE",
        "http://125.227.50.167:4444/CMoneyAdox/AdoxcService.svc",
    ).rstrip("/")


def query_table(
    format_sql: str,
    table_names: list[str],
    timeout: float = 30.0,
    session: Optional[requests.Session] = None,
) -> list[dict[str, Any]]:
    """Call /QueryTable and decode the nested JSON ResultValue."""
    url = f"{_get_base_url()}/QueryTable"
    payload = {"FormatSQL": format_sql, "TableNames": table_names}
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
    }

    s = session or requests.Session()
    resp = s.post(
        url,
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    err = data.get("Error", "")
    if err:
        raise RuntimeError(f"Adoxc QueryTable error: {err}")

    result_value = data.get("ResultValue", "")
    if not isinstance(result_value, str):
        raise ValueError("Unexpected ResultValue type from QueryTable")

    try:
        rows = json.loads(result_value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to decode ResultValue JSON: {exc}"
        ) from exc

    if not isinstance(rows, list):
        raise ValueError("ResultValue JSON is not a list")
    return rows


def query_table_light(
    format_sql: str,
    table_names: list[str],
    timeout: float = 30.0,
    session: Optional[requests.Session] = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Call /QueryTableLight and parse its ^-separated CSV response."""
    url = f"{_get_base_url()}/QueryTableLight"
    payload = {"FormatSQL": format_sql, "TableNames": table_names}
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
    }

    s = session or requests.Session()
    resp = s.post(
        url,
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    err = data.get("Error", "")
    if err:
        raise RuntimeError(f"Adoxc QueryTableLight error: {err}")

    result_value = data.get("ResultValue", "")
    if not isinstance(result_value, str):
        raise ValueError("Unexpected ResultValue type from QueryTableLight")

    text = result_value.replace("\\u000d\\u000a", "\r\n")
    reader = csv.reader(io.StringIO(text), delimiter="^")

    try:
        headers_row = next(reader)
    except StopIteration:
        return [], []

    rows: list[dict[str, Any]] = []
    for row in reader:
        if not row or all(cell == "" for cell in row):
            continue
        rows.append(dict(zip(headers_row, row)))

    return rows, headers_row
