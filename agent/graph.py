"""
agent/graph.py

构建并编译 LangGraph Agent 图。
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.nodes import orchestrator_node, tools_node_with_state_update, should_continue


def build_graph():
    """
    构建 Agent 图：
      START → orchestrator → (有工具调用) → tools → orchestrator → ...
                           → (无工具调用或达到上限) → END
    """
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("tools",        tools_node_with_state_update)

    # 起点 → orchestrator
    graph.add_edge(START, "orchestrator")

    # orchestrator → 条件路由
    graph.add_conditional_edges(
        "orchestrator",
        should_continue,
        {"tools": "tools", "end": END},
    )

    # tools → 回到 orchestrator（形成循环）
    graph.add_edge("tools", "orchestrator")

    return graph.compile()


# 模块级单例，避免重复编译
agent_graph = build_graph()
