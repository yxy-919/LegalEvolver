from __future__ import annotations

import json
from typing import Any, Dict, List


SENTENCE_INTERVAL_CANDIDATES = [
    (0, "二十年以上有期徒刑、无期徒刑或者死刑"),
    (1, "十年以上二十年以下有期徒刑"),
    (2, "七年以上十年以下有期徒刑"),
    (3, "五年以上七年以下有期徒刑"),
    (4, "三年以上五年以下有期徒刑"),
    (5, "二年以上三年以下有期徒刑"),
    (6, "一年以上二年以下有期徒刑"),
    (7, "九个月以上一年以下有期徒刑"),
    (8, "六个月以上九个月以下有期徒刑"),
    (9, "六个月以下有期徒刑"),
    (10, "无刑罚"),
]


def available_tools() -> str:
    tool_list: List[Dict[str, Any]] = [
        {
            "tool_name": "statute_search",
            "desc": "（法条检索）给定查询，返回topk最相关的法条。",
            "args": {"tool_name": "statute_search", "query": "str", "topk": "int"},
        },
        {
            "tool_name": "case_search",
            "desc": "（类案检索）给定查询，返回topk最相似的类案。",
            "args": {"tool_name": "case_search", "query": "str", "topk": "int"},
        },
        {
            "tool_name": "guideline_search",
            "desc": "（量刑指导意见检索）给定查询，返回topk最相关的量刑指导意见。",
            "args": {"tool_name": "guideline_search", "query": "str", "topk": "int"},
        },
        {
            "tool_name": "judicial_interpretation_search",
            "desc": "（司法解释检索）给定查询，返回topk最相关的司法解释条款。",
            "args": {
                "tool_name": "judicial_interpretation_search",
                "query": "str",
                "topk": "int",
            },
        },
    ]
    return _format_tool_list(tool_list)


def _format_tool_list(tool_list: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for tool in tool_list:
        args = json.dumps(tool["args"], ensure_ascii=False)
        lines.append(
            f"- {tool['tool_name']}: {tool['desc']}\n"
            f"  参数: {args}"
        )
    return "\n".join(lines)



def _sentence_interval_text() -> str:
    lines = ["刑期区间类别只能从以下候选项中选择一个，最终 JSON 中只填写类别编号："]
    lines.extend(f"{label}: {interval}" for label, interval in SENTENCE_INTERVAL_CANDIDATES)
    return "\n".join(lines)


def _single_defendant_output_schema() -> str:
    schema = {
        "relevant_articles": ["第*条"],
        "accusation": "*罪",
        "sentence_interval": 0,
    }
    return json.dumps(schema, ensure_ascii=False, indent=2)


def _multi_defendant_output_schema(defendants: List[str]) -> str:
    schema = {
        "defendants": [
            {
                "name": defendant,
                "judgments": [
                    {
                        "relevant_articles": ["第*条"],
                        "accusation": "*罪",
                        "sentence_interval": 0,
                    }
                ],
            }
            for defendant in defendants
        ]
    }
    return json.dumps(schema, ensure_ascii=False, indent=2)


def get_instructions(max_search_limit: int = 10) -> str:
    return f"""你是一名严谨的刑事法律判决预测Agent。你需要在可调用检索工具的环境中，根据给定案情事实完成相关法条预测、罪名预测和刑期区间预测。

## 1. 核心任务
你需要输出一个完整的推理轨迹，并在最终给出结构化答案：
- 对单被告案件：预测该被告人的相关刑法条文、罪名和刑期区间类别。
- 对多被告案件：分别预测每名被告人的每个判项，包括相关刑法条文、罪名和刑期区间类别。

预测目标包括：
1. **relevant_articles**：刑法条文list。
2. **accusation**：标准罪名。
3. **sentence_interval**：
{_sentence_interval_text()}
只输出 0 到 10 的类别编号，不输出具体刑期。

## 2. 推理工作流
你应当按照下面的循环完成任务：

1. **分析推理**：使用 `<think>``</think>` 标签规划步骤，并写出推理过程。
2. **行动**：当需要补充法条、类似案件、量刑指导意见、司法解释时，使用 `<tool_call></tool_call>` 发起工具调用。
3. **观察**：工具调用后，你会收到 `<tool_call_result></tool_call_result>`，请分析返回的检索结果。
4. **迭代**：重复上面的“分析推理-行动-观察”过程，每一轮迭代都应基于上一轮观察推进结论。
5. **完成**：当你已经有足够依据支持最终预测时，停止工具调用。将判决预测的结果放入 `<answer></answer>`标签中，标签内部必须是合法 JSON；最终 JSON 之外不要输出额外解释。

## 3. 可使用工具
你最多可以进行 {max_search_limit} 次工具调用，所以当法条、罪名边界、司法解释标准或量刑区间不确定时，应优先使用工具获取准确的法律知识。

{available_tools()}

## 4. 工具调用格式
当需要调用工具时，严格输出以下格式：

<tool_call>{{"tool_name": "工具名", "query": "查询内容", "topk": 返回条数}}</tool_call>

工具调用结果会由环境以以下格式返回：

<tool_call_result>工具调用的结果</tool_call_result>

"""


def get_single_defendant_user_prompt(fact: str, criminal: str) -> str:
    return f"""## 案件类型
单被告刑事案件

## 预测对象
被告人：{criminal}

## 案情事实
{fact}

## 本案要求
请为上述被告人预测相关法条、罪名和刑期区间类别。只有一个相关法条和罪名。

## 最终答案 JSON Schema
最终必须在 `<answer></answer>` 标签内部输出如下 JSON 结构：
{_single_defendant_output_schema()}
"""


def get_multi_defendant_user_prompt(fact: str, defendants: List[str]) -> str:
    defendant_text = "、".join(defendants) if defendants else "未提供具体姓名的被告人"
    return f"""## 案件类型
多被告刑事案件

## 预测对象
请分别预测以下被告人的判决结果：{defendant_text}

## 案情事实
{fact}

## 本案要求
请逐名分析每名被告人的罪名、相关法条和刑期区间类别。同一被告人可能涉及一个或多个罪名；如果涉及多个罪名，应在该被告人的 `judgments` 中分别输出。

## 最终答案 JSON Schema
最终必须在 `<answer></answer>` 标签内部输出如下 JSON 结构：
{_multi_defendant_output_schema(defendants)}
"""


def get_user_prompt(
    fact: str,
    criminal_or_defendants: str | List[str],
    dataset_type: str = "single",
) -> str:
    if dataset_type == "multi":
        if isinstance(criminal_or_defendants, str):
            defendants = [criminal_or_defendants]
        else:
            defendants = [str(item) for item in criminal_or_defendants]
        return get_multi_defendant_user_prompt(fact, defendants)

    if isinstance(criminal_or_defendants, list):
        criminal = str(criminal_or_defendants[0]) if criminal_or_defendants else "被告人"
    else:
        criminal = str(criminal_or_defendants)
    return get_single_defendant_user_prompt(fact, criminal)
