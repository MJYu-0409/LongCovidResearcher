"""
main.py - 项目入口
运行方式：python main.py
"""

from __future__ import annotations

import argparse

from infra import configure_logging

configure_logging()

import os

from agent import run as run_agent
from storage.postgres.session_store import load as load_session, save as save_session
from agent.summarizer import run_summarizer
from data_pipeline.pipeline import run as run_pipeline

# 每轮结束后保留的最近消息条数，其余压入摘要
RECENT_MESSAGES_KEEP = 3
# 跨会话记忆：持久化用的 session_id，可通过环境变量覆盖
AGENT_SESSION_ID = os.getenv("AGENT_SESSION_ID", "default")


def _run_agent_cli() -> int:
    """
    交互式 Agent CLI（默认入口）。
    - 自动保留多轮对话（摘要 + 最近 N 条）；每轮结束立即持久化到 PostgreSQL，断线/异常最多丢当前轮
    - 启动时按 AGENT_SESSION_ID 加载上次会话，实现跨会话记忆
    """
    print("=" * 68)
    print("Long COVID Researcher - Agent 交互模式")
    print("提示：输入 exit/quit 退出")
    print("=" * 68)

    # 跨会话：尝试加载上次持久化的状态
    loaded = load_session(AGENT_SESSION_ID)
    if loaded:
        summary = loaded.get("summary", "") or ""
        history = loaded.get("history")
        retrieved_chunks = loaded.get("retrieved_chunks")
    else:
        summary = ""
        history = None
        retrieved_chunks = None

    while True:
        user_input = input("\n你：").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            return 0

        result = run_agent(
            user_input,
            history=history,
            retrieved_chunks=retrieved_chunks,
            summary=summary or None,
        )
        print("\n助手：")
        print(result.get("answer", ""))

        # 本轮结束：摘要 + 保留最近 N 条，并立即持久化
        summary, history = run_summarizer(
            result.get("summary", ""),
            result.get("messages", []),
            keep_last_n=RECENT_MESSAGES_KEEP,
        )
        retrieved_chunks = result.get("retrieved_chunks")
        save_session(AGENT_SESSION_ID, summary, history, retrieved_chunks or [])


def main() -> int:
    parser = argparse.ArgumentParser(description="Long COVID Researcher 项目入口")
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="运行数据流水线（默认行为以 data_pipeline.pipeline.run() 为准）",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="启动 FastAPI 服务（Agent 对话、检索、健康检查），默认端口 8001",
    )
    try:
        _default_port = int(os.getenv("API_PORT", "8001"))
    except (TypeError, ValueError):
        _default_port = 8001
    parser.add_argument(
        "--port",
        type=int,
        default=_default_port,
        help="FastAPI 监听端口（仅 --api 时生效），默认 API_PORT 或 8001",
    )
    args = parser.parse_args()

    if args.pipeline:
        run_pipeline()
        return 0

    if args.api:
        import uvicorn
        uvicorn.run("api.app:app", host="0.0.0.0", port=args.port, reload=False)
        return 0

    return _run_agent_cli()


if __name__ == "__main__":
    raise SystemExit(main())
