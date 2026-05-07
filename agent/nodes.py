"""
agent/nodes.py

LangGraph 节点定义：
  - orchestrator_node：Qwen 决策，选择工具或结束
  - tools_node：执行工具调用，并把检索结果写入 State
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

_QA_TOOLS = {"answer_question", "synthesize_review"}


def _inject_full_chunks(last_msg: AIMessage, chunks: list[dict]) -> AIMessage:
    """
    Replace the context_chunks arg for answer_question / synthesize_review with
    the full payload text (including context_text from ±1 neighbor expansion)
    stored in State.  The LLM only sees 150-char previews from search_literature,
    so without this injection those tools would generate answers from truncated text.
    """
    if not chunks or not last_msg.tool_calls or not any(tc["name"] in _QA_TOOLS for tc in last_msg.tool_calls):
        return last_msg

    chunks_payload = []
    for c in chunks[:20]:
        p = c.get("payload", c)
        chunks_payload.append({
            "pmcid":    p.get("pmcid", ""),
            "section":  p.get("section", ""),
            "text":     p.get("context_text") or p.get("text", ""),
            "pub_year": p.get("pub_year", ""),
        })
    chunks_json = json.dumps(chunks_payload, ensure_ascii=False)

    new_tool_calls = [
        {**tc, "args": {**tc.get("args", {}), "context_chunks": chunks_json}}
        if tc["name"] in _QA_TOOLS else tc
        for tc in last_msg.tool_calls
    ]
    return AIMessage(content=last_msg.content or "", tool_calls=new_tool_calls, id=last_msg.id)

from agent.state import AgentState
from agent.tools import ALL_TOOLS
from infra.clients import get_qwen_chat_model

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5

# Orchestrator：Qwen 单例（从 infra 统一入口获取）
_orchestrator_llm = get_qwen_chat_model(temperature=0, max_tokens=2000).bind_tools(ALL_TOOLS)

_SYSTEM_PROMPT = """你是 Long COVID 学术研究助手，帮助研究人员、政策制定者和学生分析学术文献。

你有以下工具：
- search_literature：检索相关文献（优先第一步使用）
- get_paper_detail：获取某篇论文的完整信息
- analyze_sentiment：分析学界对某议题的情感态度
- answer_question：基于已检索文献回答具体问题
- synthesize_review：综合多篇文献生成综述（信息充分后才使用）

工作原则：
1. 收到问题后，先用 search_literature 检索相关文献；传入 query 时将用户问题翻译为英文以提升检索效果
2. 事实性问题（具体数据、结论）→ answer_question
3. 综合性问题（研究现状、系统梳理）→ 多轮检索后 synthesize_review
4. 情感/态度问题 → analyze_sentiment
5. 最多进行 {max_iter} 轮工具调用，然后给出最终答案
6. 最终答案用中文，引用来源注明 pmcid；末尾必须附一行：[证据充分度：高/中/低，基于 N 篇文献]
7. 【严格禁止】若 search_literature 返回 error 字段或 count 为 0 且含错误信息，必须直接告知用户检索系统不可用，绝对不能用自身训练知识代替文献回答——本系统的价值在于文献溯源，脱离文献的回答会误导用户
8. 若 search_literature 返回的 count < 3，必须换关键词或同义词重新检索（最多重试2次），确认获得 ≥3 篇相关文献后再调用 answer_question；不得在文献不足时强行作答
9. 调用 synthesize_review 前确保已累积 ≥5 个 chunk；若不足，继续检索补充后再综述
10. 调用 answer_question 或 synthesize_review 时，若此前对话摘要中记录了用户的专业背景、研究方向或偏好（如"临床医生"、"偏好机制分析"、"不要列举统计数字"），须将这些信息折叠进 question/topic 参数，使问题更精确——例如在问题末尾追加 "Focus on [角度], suitable for [背景]"。不得将记忆内容作为独立参数传入，不得将其作为事实来源引用""".format(max_iter=MAX_ITERATIONS)


def orchestrator_node(state: AgentState) -> dict:
    """
    Orchestrator 节点：调用 Qwen 决定下一步行动（调用工具或给出最终答案）。
    """
    from langchain_core.messages import SystemMessage

    messages = state["messages"]

    # 注入系统提示；若有此前对话摘要则一并注入，供模型利用
    if not any(m.type == "system" for m in messages):
        system_parts = [SystemMessage(content=_SYSTEM_PROMPT)]
        if state.get("summary"):
            system_parts.append(SystemMessage(content="【此前对话摘要】\n" + state["summary"]))
        messages = system_parts + list(messages)

    response = _orchestrator_llm.invoke(messages)
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    logger.info("orchestrator: iteration=%d, tool_calls=%d",
                state.get("iteration_count", 0), len(tool_calls))
    for tc in tool_calls:
        args_preview = {k: str(v)[:80] for k, v in (tc.get("args") or {}).items()}
        logger.info("  → tool: %s  args: %s", tc.get("name"), args_preview)

    return {
        "messages":       [response],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def tools_node_with_state_update(state: AgentState) -> dict:
    """
    工具执行节点：执行所有工具调用，并将 search_literature 的完整 chunk 写入 State。
    """
    messages  = state["messages"]
    last_msg  = messages[-1]

    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return {}

    # Inject full chunks into answer_question / synthesize_review before execution.
    # The LLM constructs context_chunks from 150-char search previews; we replace
    # that with the full context_text (±1 neighbor expanded) stored in State.
    full_chunks = list(state.get("retrieved_chunks", []))
    invoke_state = state
    if full_chunks:
        patched = _inject_full_chunks(last_msg, full_chunks)
        if patched is not last_msg:
            invoke_state = {**state, "messages": list(state["messages"][:-1]) + [patched]}

    # 先用 LangGraph 内置 ToolNode 执行工具
    tool_node = ToolNode(ALL_TOOLS)
    try:
        result = tool_node.invoke(invoke_state)
    except Exception as exc:
        logger.error("ToolNode.invoke failed: %s", exc)
        # 返回空更新，让 Orchestrator 收到空结果后按 prompt 拒绝回答
        return {"retrieved_chunks": full_chunks}

    # 提取 search_literature 的完整 chunks 写入 State（从 ContextVar 读，并发安全）
    new_chunks = full_chunks

    has_search_call = any(tc["name"] == "search_literature" for tc in last_msg.tool_calls)
    if has_search_call:
        from agent.tools.search import _last_chunks_var
        cached = _last_chunks_var.get([])
        if cached:
            new_chunks = new_chunks + cached

    return {
        **result,
        "retrieved_chunks": new_chunks,
    }


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    路由函数：判断是继续调用工具还是结束。
    """
    last_msg = state["messages"][-1]
    iteration = state.get("iteration_count", 0)

    # 达到最大迭代数，强制结束
    if iteration >= MAX_ITERATIONS:
        logger.info("达到最大迭代数 %d，结束", MAX_ITERATIONS)
        return "end"

    # Orchestrator 没有发起工具调用，说明已得出答案
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return "end"

    return "tools"
