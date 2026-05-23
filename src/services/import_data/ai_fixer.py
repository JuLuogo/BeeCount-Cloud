"""AI 自动纠错 —— 导入账单解析失败时,用 LLM 推断修复。

场景:
- tx_type 字段值为"不计收支"等无法识别的类型
- 金额格式异常(如含中文数字、特殊货币符号)
- 时间格式无法解析
- 分类名称需要智能映射

设计原则:
- 仅修复**可明确推断**的错误,不确定时保持原样(让前端提示用户)
- 批量调用:一次 LLM 请求处理所有错误行,降低成本
- 失败降级:LLM 不可用/超时时,保持原始错误,不阻断导入流程
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..ai import call_chat_json, resolve_chat_provider
from ..ai.provider_client import ChatProviderError, JsonParseFailedError
from .schema import ImportError, ImportFieldMapping, ParsedRow

if TYPE_CHECKING:
    from ...models import User, UserProfile

logger = logging.getLogger(__name__)


@dataclass
class AIFixResult:
    """单行的 AI 修复结果。"""

    row_number: int
    field_name: str
    original_value: str
    fixed_value: str
    confidence: str  # "high" | "medium" | "low"
    reason: str


@dataclass
class AIFixSummary:
    """AI 纠错汇总。"""

    fixes: list[AIFixResult]
    fixable_error_indices: list[int]  # 对应原始 errors list 的索引
    unfixable_errors: list[ImportError]  # AI 无法修复的错误


_FIX_TX_TYPE_PROMPT_ZH = """\
你是 BeeCount(蜜蜂记账)的数据清洗助手。用户导入账单时,部分行的"收支类型"字段无法识别,需要你推断正确的类型。

已知规则:
- "expense"(支出):消费、付款、转出、扣款、还款、费用、手续费、不计收支(如果是支出场景)
- "income"(收入):工资、奖金、退款、转入、收款、收益、红包(收入场景)
- "transfer"(转账):账户间转账、换汇、信用卡还款(如果是自己账户间转移)

请对下面每行无法识别的类型值进行判断,返回 JSON:
{{
  "fixes": [
    {{
      "row_number": 行号,
      "field_name": "tx_type",
      "original_value": "原始值",
      "fixed_value": "expense|income|transfer|null",
      "confidence": "high|medium|low",
      "reason": "判断理由"
    }}
  ]
}}

注意:
1. 如果完全无法判断,fixed_value 填 null
2. "不计收支"通常是 expense(如手续费、保险),也可能是 transfer(如信用卡还款)
3. 只输出 JSON,不要任何解释文字

数据:
{rows}
"""


async def try_fix_import_errors(
    *,
    errors: list[ImportError],
    rows: list[ParsedRow],
    mapping: ImportFieldMapping,
    user: User,
    profile: UserProfile | None,
) -> AIFixSummary:
    """尝试用 AI 修复导入错误。

    - 仅处理 tx_type 相关的 PARSE_INVALID_FIELD 错误(如截图中的"不计收支")
    - 一次 LLM 调用批量处理
    - LLM 失败时降级返回空修复
    """
    fixes: list[AIFixResult] = []
    fixable_indices: list[int] = []
    unfixable: list[ImportError] = []

    # 1. 筛选 AI 可能修复的错误(目前只处理 tx_type)
    tx_type_errors: list[tuple[int, ImportError]] = []
    for idx, err in enumerate(errors):
        if err.code == "PARSE_INVALID_FIELD" and err.field_name == "tx_type":
            tx_type_errors.append((idx, err))
        else:
            unfixable.append(err)

    if not tx_type_errors:
        return AIFixSummary(
            fixes=fixes,
            fixable_error_indices=fixable_indices,
            unfixable_errors=unfixable,
        )

    # 2. 构建 prompt 数据
    error_rows_data = []
    for idx, err in tx_type_errors:
        # 找到对应行的原始数据
        row = next((r for r in rows if r.row_number == err.row_number), None)
        if row is None:
            unfixable.append(err)
            continue
        tx_type_col = mapping.tx_type or "tx_type"
        raw_type = row.cells.get(tx_type_col, "")
        error_rows_data.append({
            "row_number": err.row_number,
            "tx_type_value": raw_type,
            "raw_line": err.raw_line[:200],
        })

    if not error_rows_data:
        return AIFixSummary(
            fixes=fixes,
            fixable_error_indices=fixable_indices,
            unfixable_errors=unfixable,
        )

    # 3. 调用 LLM
    try:
        provider = resolve_chat_provider(user, profile)
    except ChatProviderError as exc:
        logger.warning("ai_fixer no provider: %s", exc)
        # 降级:所有错误保持原样
        return AIFixSummary(
            fixes=fixes,
            fixable_error_indices=fixable_indices,
            unfixable_errors=errors,
        )

    prompt = _FIX_TX_TYPE_PROMPT_ZH.format(
        rows=json.dumps(error_rows_data, ensure_ascii=False, indent=2)
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        result = await call_chat_json(
            config=provider,
            messages=messages,
            timeout=30.0,
            max_retries=1,
        )
    except (ChatProviderError, JsonParseFailedError) as exc:
        logger.warning("ai_fixer llm failed: %s", exc)
        return AIFixSummary(
            fixes=fixes,
            fixable_error_indices=fixable_indices,
            unfixable_errors=errors,
        )

    # 4. 解析 LLM 返回
    fixes_list = []
    if isinstance(result, dict):
        fixes_list = result.get("fixes", [])
    elif isinstance(result, list) and len(result) > 0:
        fixes_list = result[0].get("fixes", []) if isinstance(result[0], dict) else []

    fixed_row_numbers: set[int] = set()
    for fix_item in fixes_list:
        if not isinstance(fix_item, dict):
            continue
        row_num = fix_item.get("row_number")
        fixed_value = fix_item.get("fixed_value")
        if fixed_value is None or fixed_value == "null":
            continue
        if fixed_value not in ("expense", "income", "transfer"):
            continue

        # 找到对应的 error index
        for idx, err in tx_type_errors:
            if err.row_number == row_num and row_num not in fixed_row_numbers:
                fixed_row_numbers.add(row_num)
                fixable_indices.append(idx)
                fixes.append(
                    AIFixResult(
                        row_number=row_num,
                        field_name="tx_type",
                        original_value=fix_item.get("original_value", ""),
                        fixed_value=fixed_value,
                        confidence=fix_item.get("confidence", "medium"),
                        reason=fix_item.get("reason", ""),
                    )
                )
                break

    # 未被 AI 修复的错误加入 unfixable
    fixed_idx_set = set(fixable_indices)
    for idx, err in tx_type_errors:
        if idx not in fixed_idx_set:
            unfixable.append(err)

    return AIFixSummary(
        fixes=fixes,
        fixable_error_indices=fixable_indices,
        unfixable_errors=unfixable,
    )


def apply_ai_fixes_to_rows(
    *,
    rows: list[ParsedRow],
    fixes: list[AIFixResult],
    mapping: ImportFieldMapping,
) -> list[ParsedRow]:
    """将 AI 修复应用到 ParsedRow 列表,返回修复后的新列表。"""
    fix_map = {f.row_number: f for f in fixes}
    new_rows: list[ParsedRow] = []

    for row in rows:
        fix = fix_map.get(row.row_number)
        if fix is None or fix.field_name != "tx_type":
            new_rows.append(row)
            continue

        # 创建新的 cells dict,修改 tx_type 值
        new_cells = dict(row.cells)
        tx_type_col = mapping.tx_type or "tx_type"
        new_cells[tx_type_col] = fix.fixed_value
        new_rows.append(
            ParsedRow(
                row_number=row.row_number,
                cells=new_cells,
                raw_line=row.raw_line,
            )
        )

    return new_rows
