"""
agent/nodes.py

LangGraph 节点定义：
  - orchestrator_node：GPT-4o 决策，选择工具或结束
  - tools_node：执行工具调用，并把检索结果写入 State
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.tools import ALL_TOOLS
from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5

# GPT-4o Orchestrator，绑定所有工具
_orchestrator_llm = ChatOpenAI(
    model="gpt-4o",
    api_key=OPENAI_API_KEY,
    temperature=0,
    max_tokens=1000,
).bind_tools(ALL_TOOLS)

_SYSTEM_PROMPT = """你是 Long COVID 学术研究助手，帮助研究人员、政策制定者和学生分析学术文献。

你有以下工具：
- search_literature：检索相关文献（优先第一步使用）
- get_paper_detail：获取某篇论文的完整信息
- analyze_sentiment：分析学界对某议题的情感态度
- answer_question：基于已检索文献回答具体问题
- synthesize_review：综合多篇文献生成综述（信息充分后才使用）

工作原则：
1. 收到问题后，先用 search_literature 检索相关文献
2. 事实性问题（具体数据、结论）→ answer_question
3. 综合性问题（研究现状、系统梳理）→ 多轮检索后 synthesize_review
4. 情感/态度问题 → analyze_sentiment
5. 最多进行 {max_iter} 轮工具调用，然后给出最终答案
6. 最终答案用中文，引用来源注明 pmcid""".format(max_iter=MAX_ITERATIONS)


def orchestrator_node(state: AgentState) -> dict:
    """
    Orchestrator 节点：调用 GPT-4o 决定下一步行动（调用工具或给出最终答案）。
    """
    messages = state["messages"]

    # 第一条消息注入系统提示
    if not any(m.type == "system" for m in messages):
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=_SYSTEM_PROMPT)] + list(messages)

    response = _orchestrator_llm.invoke(messages)
    logger.info("orchestrator: iteration=%d, tool_calls=%d",
                state.get("iteration_count", 0),
                len(response.tool_calls) if hasattr(response, "tool_calls") else 0)

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

    # 先用 LangGraph 内置 ToolNode 执行工具
    tool_node = ToolNode(ALL_TOOLS)
    result    = tool_node.invoke(state)

    # 提取 search_literature 的完整 chunks 写入 State
    new_chunks = list(state.get("retrieved_chunks", []))

    for tool_call in last_msg.tool_calls:
        if tool_call["name"] != "search_literature":
            continue

        # 从工具调用参数重跑一次，拿到完整 chunks
        from agent.tools.search import search_literature_fn
        args = tool_call.get("args", {})
        try:
            full_result = search_literature_fn(
                query    = args.get("query", ""),
                top_n    = args.get("top_n", 10),
                pub_year = args.get("pub_year"),
                journal  = args.get("journal"),
            )
            new_chunks.extend(full_result["chunks"])
        except Exception as e:
            logger.warning("获取完整 chunks 失败: %s", e)

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
