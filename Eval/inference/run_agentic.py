from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .prompts import get_instructions, get_user_prompt

BEGIN_TOOL_CALL = "<tool_call>"
END_TOOL_CALL = "</tool_call>"
BEGIN_TOOL_RESULT = "<tool_call_result>"
END_TOOL_RESULT = "</tool_call_result>"
DEBUG_TOOL_LOGS = os.environ.get("REACT_DEBUG_TOOL_LOGS") == "1"
TOOLS = None


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def load_retrieval_tools():
    from Retrieval._base_retriever import get_shared_encoder
    from Retrieval.case_retriever import case_search
    from Retrieval.guideline_retriever import guideline_search
    from Retrieval.judicial_interpretation_retriever import (
        judicial_interpretation_search,
    )
    from Retrieval.statute_retriever import statute_search

    return {
        "case_search": case_search,
        "get_shared_encoder": get_shared_encoder,
        "guideline_search": guideline_search,
        "judicial_interpretation_search": judicial_interpretation_search,
        "statute_search": statute_search,
    }

def get_tools() -> Dict[str, Any]:
    global TOOLS
    if TOOLS is None:
        loaded = load_retrieval_tools()
        TOOLS = {
            "statute_search": loaded["statute_search"],
            "case_search": loaded["case_search"],
            "guideline_search": loaded["guideline_search"],
            "judicial_interpretation_search": loaded["judicial_interpretation_search"],
        }
    return TOOLS

def prewarm_retrieval() -> None:
    loaded = load_retrieval_tools()
    tools = get_tools()
    print("Prewarming SAILER encoder and retrieval indexes...")
    loaded["get_shared_encoder"]().encode(["检索预热"])
    for tool_name, query in [
        ("statute_search", "故意伤害罪 刑法"),
        ("case_search", "故意伤害 轻伤"),
        ("guideline_search", "故意伤害罪 轻伤 量刑"),
        ("judicial_interpretation_search", "故意伤害罪 司法解释"),
    ]:
        try:
            tools[tool_name](query, 1)
        except Exception as exc:
            print(f"Prewarm warning for {tool_name}: {type(exc).__name__}: {exc}")

def load_completed_counts(output_path: Path) -> Dict[str, int]:
    completed_counts: Dict[str, int] = {}
    if not output_path.exists():
        return completed_counts
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            case_id = row.get("caseID")
            if case_id is None:
                continue
            case_id = str(case_id)
            completed_counts[case_id] = completed_counts.get(case_id, 0) + 1
    return completed_counts

def sequence_to_output_row(seq: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "caseID": seq["item"].get("caseID"),
        "sample_idx": seq["sample_idx"],
        "dataset_type": seq["dataset_type"],
        "predict": f"{seq['output']}",
        "tool_call": seq["tool_call"],
    }
    if seq["dataset_type"] == "multi":
        row["defendants"] = seq["item"].get("defendants", [])
    else:
        row["criminal"] = ((seq["item"].get("meta") or {}).get("criminals") or [None])[0]
    return row

def _plain_card_texts(cards: Iterable[Dict[str, Any]], *, limit: int = 10) -> List[str]:
    texts: List[str] = []
    for idx, card in enumerate(cards, start=1):
        if idx > limit:
            break
        text = str(card.get("text") or "").strip()
        metadata = card.get("metadata") if isinstance(card.get("metadata"), dict) else {}
        if card.get("source_type") == "precedent_case":
            accusation = "、".join(
                str(item) for item in metadata.get("accusation", []) if str(item).strip()
            )
            articles = "、".join(
                str(item)
                for item in metadata.get("relevant_articles", [])
                if str(item).strip()
            )
            imprisonment = metadata.get("imprisonment")
            outcome_parts = []
            if accusation:
                outcome_parts.append(f"罪名：{accusation}")
            if articles:
                outcome_parts.append(f"相关法条：{articles}")
            if imprisonment is not None:
                outcome_parts.append(f"刑期：{imprisonment}个月")
            if outcome_parts:
                text = f"{text}\n裁判结果：{'；'.join(outcome_parts)}。"
        if text:
            texts.append(text)
    return texts

def _truncate_text(text: str, max_chars: int = 4000) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    head = max_chars * 3 // 4
    tail = max_chars - head
    return text[:head] + "\n\n[检索结果过长，已省略中间部分。]\n\n" + text[-tail:]

def observation_text(tool_name: str, tool_result: Dict[str, Any]) -> str:
    parts: List[str] = []
    if tool_name == "guideline_search":
        context = tool_result.get("sentencing_general_context")
        if isinstance(context, dict) and context.get("text"):
            parts.append(str(context["text"]).strip())
        parts.extend(_plain_card_texts(tool_result.get("national_guideline_evidence", [])))
        parts.extend(_plain_card_texts(tool_result.get("local_guideline_evidence", [])))
    else:
        parts.extend(_plain_card_texts(tool_result.get("results", [])))
    if tool_result.get("error"):
        parts.append(f"工具调用失败：{tool_result['error']}")
    if not parts:
        parts.append("未检索到相关内容。")
    return _truncate_text("\n\n".join(part for part in parts if part), 4000)

def execute_tool(tool_name: str, tool_query: str, tool_topk: int) -> Dict[str, Any]:
    tools = get_tools()
    if tool_name not in tools:
        result = {
            "tool_name": tool_name,
            "query": tool_query,
            "results": [],
            "error": f"unknown tool: {tool_name}",
        }
        return {
            "observation": observation_text(tool_name, result),
            "raw_result": result,
        }

    result = tools[tool_name](tool_query, tool_topk)
    return {
        "observation": observation_text(tool_name, result),
        "raw_result": result,
    }

async def generate_response(
    client: Any,
    prompt: str,
    semaphore: asyncio.Semaphore,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float | None,
    top_k: int,
    min_p: float,
    model_name: str,
    stop_on_tool_call: bool = True,
    retry_limit: int = 3,
) -> str:
    extra_body: Dict[str, Any] = {
        "top_k": top_k,
        "min_p": min_p,
    }
    stop = None
    if stop_on_tool_call:
        stop = [END_TOOL_CALL]
        extra_body["include_stop_str_in_output"] = True
    if repetition_penalty is not None:
        extra_body["repetition_penalty"] = repetition_penalty

    for attempt in range(retry_limit):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=min(max_tokens, 32768),
                    stop=stop,
                    extra_body=extra_body,
                    timeout=600,
                )
                return response.choices[0].message.content or ""
        except Exception as exc:
            if attempt == retry_limit - 1:
                print(f"Failed after {retry_limit} attempts: {exc}")
                return ""
            await asyncio.sleep(1 * (attempt + 1))
    return ""

def extract_between(text: str, start_marker: str, end_marker: str) -> str | None:
    try:
        pattern = re.escape(end_marker[::-1]) + r"(.*?)" + re.escape(start_marker[::-1])
        matches = re.findall(pattern, text[::-1], flags=re.DOTALL)
        if matches:
            return matches[0][::-1].strip()
        return None
    except Exception as exc:
        print(f"---Error:---\n{exc}")
        print("-------------------")
        return None

async def process_single_sequence(
    seq: Dict[str, Any],
    client: Any,
    semaphore: asyncio.Semaphore,
    tool_semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    turn: int = 0,
) -> Dict[str, Any]:
    async def force_final_answer() -> None:
        # if "<answer>" in seq["output"]:
        #     return
        final_prompt = (
            "\n\n请停止调用工具。请只基于上面的案情、推理和工具结果给出最终预测。"
            "最终答案必须放在 `<answer></answer>` 标签内，标签内部必须是合法 JSON；"
            "不要在最终 JSON 外输出解释。"
        )
        seq["prompt"] += final_prompt
        seq["output"] += final_prompt
        final_text = await generate_response(
            client=client,
            prompt=seq["prompt"],
            semaphore=semaphore,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k_sampling,
            min_p=args.min_p,
            model_name=args.model_name,
            stop_on_tool_call=False,
        )
        seq["prompt"] += final_text
        seq["output"] += final_text

    while not seq["finished"]:
        text = await generate_response(
            client=client,
            prompt=seq["prompt"],
            semaphore=semaphore,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k_sampling,
            min_p=args.min_p,
            model_name=args.model_name,
            stop_on_tool_call=True,
        )

        seq["prompt"] += text
        seq["output"] += text

        if turn >= args.max_turn:
            await force_final_answer()
            seq["finished"] = True
            break

        last_open = text.rfind(BEGIN_TOOL_CALL)
        last_close = text.rfind(END_TOOL_CALL)

        # 没有工具调用，说明模型自然给出了最终答案或自然结束
        if last_open < 0:
            # await force_final_answer()
            seq["finished"] = True
            break

        # 有闭合标签，但闭合标签在 open 之前，属于异常格式
        if last_close >= 0 and last_close < last_open:
            # await force_final_answer()
            seq["finished"] = True
            break

        # 有 <tool_call> 但没有后续 </tool_call>
        if last_close < last_open:
            tool_tail = text[last_open + len(BEGIN_TOOL_CALL):].strip()

            try:
                json.loads(tool_tail)
            except Exception:
                # JSON 本身不完整，不应该强行执行工具
                # await force_final_answer()
                seq["finished"] = True
                break

            # JSON 是完整的，只是缺少闭合标签，可以补上
            seq["prompt"] += END_TOOL_CALL
            seq["output"] += END_TOOL_CALL
            text += END_TOOL_CALL

        turn += 1
        tool_call_str = extract_between(text, BEGIN_TOOL_CALL, END_TOOL_CALL)
        if DEBUG_TOOL_LOGS:
            print(f"tool_call: {tool_call_str}")
        if not tool_call_str:
            append_text = f"\n{BEGIN_TOOL_RESULT}\n请按照正确的格式调用工具。\n{END_TOOL_RESULT}\n"
            seq["prompt"] += append_text
            seq["output"] += append_text
            continue

        try:
            tool_call = json.loads(tool_call_str)
            tool_name = tool_call.get("tool_name")
            tool_query = tool_call.get("query")
            tool_topk = max(1, min(int(tool_call.get("topk") or args.retrieval_top_k), args.retrieval_top_k))
            async with tool_semaphore:
                tool_result = await asyncio.to_thread(
                    execute_tool,
                    tool_name,
                    tool_query,
                    tool_topk,
                )
            observation_preview = str(tool_result["observation"])
            if len(observation_preview) > 800:
                observation_preview = observation_preview[:800] + "\n...[truncated]"
            if DEBUG_TOOL_LOGS:
                print(f"tool_result: {observation_preview}")
            seq["search_count"] += 1
            seq.setdefault("tool_call", []).append(
                {
                    "turn": turn,
                    "tool_name": tool_name,
                    "query": tool_query,
                    "topk": tool_topk,
                    "tool_call": tool_call,
                    "observation": tool_result["observation"],
                    "raw_result": tool_result["raw_result"],
                }
            )
            append_text = (
                f"\n{BEGIN_TOOL_RESULT}\n{tool_result['observation']}\n{END_TOOL_RESULT}\n"
            )
        except Exception as exc:
            if DEBUG_TOOL_LOGS:
                print(f"error: {exc}")
            seq.setdefault("tool_call", []).append(
                {
                    "turn": turn,
                    "tool_call_raw": tool_call_str,
                    "error": str(exc),
                }
            )
            append_text = (
                f"\n{BEGIN_TOOL_RESULT}\n{exc}\n请按照正确的格式调用工具。\n{END_TOOL_RESULT}\n"
            )
        seq["prompt"] += append_text
        seq["output"] += append_text
        if turn >= args.max_turn:
            await force_final_answer()
            seq["finished"] = True
            break
    return seq

def infer_dataset_type(item: Dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    if "defendants" in item and "outcomes" in item:
        return "multi"
    return "single"

def build_prompt(item: Dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    dataset_type = infer_dataset_type(item, args.dataset_type)
    fact = item["fact"]
    instruction = get_instructions(args.max_turn)
    if dataset_type == "multi":
        defendants = [str(name) for name in item.get("defendants") or []]
        user_prompt = get_user_prompt(fact, defendants, dataset_type="multi")
    else:
        criminals = (item.get("meta") or {}).get("criminals") or ["被告人"]
        user_prompt = get_user_prompt(fact, str(criminals[0]), dataset_type="single")
    return instruction + user_prompt, dataset_type

async def main_generate_async(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI
    from tqdm import tqdm

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    completed_counts = load_completed_counts(output_path)

    clients = [
        AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=base_url,
        )
        for base_url in args.base_urls
    ]

    data = load_jsonl(args.test_dataset_path)
    prewarm_retrieval()

    active_sequences: List[Dict[str, Any]] = []
    for item in data:
        prompt, dataset_type = build_prompt(item, args)
        case_id = str(item["caseID"])
        completed_count = min(completed_counts.get(case_id, 0), args.samples_per_case)
        for sample_idx in range(completed_count, args.samples_per_case):
            active_sequences.append(
                {
                    "item": item,
                    "sample_idx": sample_idx,
                    "dataset_type": dataset_type,
                    "prompt": prompt,
                    "output": "",
                    "finished": False,
                    "search_count": 0,
                    "tool_call": [],
                }
            )

    semaphore = asyncio.Semaphore(args.concurrent_limit)
    tool_semaphore = asyncio.Semaphore(args.concurrent_limit_tool)
    write_lock = asyncio.Lock()

    async def run_and_write(seq: Dict[str, Any]) -> Dict[str, Any]:
        client = clients[seq["client_idx"] % len(clients)]
        result = await process_single_sequence(
            seq=seq,
            client=client,
            semaphore=semaphore,
            tool_semaphore=tool_semaphore,
            args=args,
        )
        row = sequence_to_output_row(result)
        async with write_lock:
            with output_path.open("a+", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
        return result

    for idx, seq in enumerate(active_sequences):
        seq["client_idx"] = idx % len(clients)

    tasks = [run_and_write(seq) for seq in active_sequences]
    with tqdm(total=len(tasks)) as pbar:

        async def track_progress(task):
            result = await task
            pbar.update(1)
            return result

        completed_sequences = await asyncio.gather(
            *[track_progress(task) for task in tasks]
        )

    search_counts = [seq["search_count"] for seq in completed_sequences]
    if search_counts:
        print(f"Average search count: {sum(search_counts) / len(search_counts)}")
    else:
        print("No pending sequences to run.")
    print(f"Output: {output_path}")
    print("Process completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LegalEvolver multi-turn inference on CAIL/CMDL JSONL.")
    parser.add_argument("--test_dataset_path", type=Path, required=True)
    parser.add_argument("--dataset_type", choices=["auto", "single", "multi"], default="auto")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_name", default="predictions.jsonl")
    parser.add_argument("--model_name", default="LegalEvolver")
    parser.add_argument("--base_urls", nargs="+", default=["http://127.0.0.1:8000/v1"])
    parser.add_argument("--retrieval_config", type=Path)
    parser.add_argument("--max_turn", type=int, default=10)
    parser.add_argument("--retrieval_top_k", type=int, default=3,
                        help="Historical checkpoint inference used at most 3 hits per call; use 4 for the paper-facing setting.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k_sampling", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--concurrent_limit", type=int, default=8)
    parser.add_argument("--concurrent_limit_tool", type=int, default=1)
    parser.add_argument("--samples_per_case", type=int, default=1)
    args = parser.parse_args()
    for field in ["max_turn", "retrieval_top_k", "max_tokens", "concurrent_limit", "concurrent_limit_tool", "samples_per_case"]:
        if getattr(args, field) < 1:
            parser.error(f"--{field} must be positive")
    if args.retrieval_config:
        os.environ["RUBRIC_REACT_CONFIG"] = str(args.retrieval_config.resolve())
    return args


def main() -> None:
    asyncio.run(main_generate_async(parse_args()))


if __name__ == "__main__":
    main()
