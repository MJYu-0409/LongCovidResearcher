"""
main.py - 项目入口
运行方式：python main.py
"""

from __future__ import annotations

import argparse

from infra import configure_logging

configure_logging()

from agent import run as run_agent
from data_pipeline.pipeline import run as run_pipeline


def _run_agent_cli() -> int:
    """
    交互式 Agent CLI（默认入口）。
    - 直接回车或输入 exit/quit 退出
    - 自动保留多轮对话 history 与 retrieved_chunks
    """
    print("=" * 68)
    print("Long COVID Researcher - Agent 交互模式")
    print("提示：输入 exit/quit 退出")
    print("=" * 68)

    history = None
    retrieved_chunks = None

    while True:
        user_input = input("\n你：").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            return 0

        result = run_agent(user_input, history=history, retrieved_chunks=retrieved_chunks)
        print("\n助手：")
        print(result.get("answer", ""))

        history = result.get("messages")
        retrieved_chunks = result.get("retrieved_chunks")


def main() -> int:
    parser = argparse.ArgumentParser(description="Long COVID Researcher 项目入口")
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="运行数据流水线（默认行为以 data_pipeline.pipeline.run() 为准）",
    )
    args = parser.parse_args()

    if args.pipeline:
        run_pipeline()
        return 0

    return _run_agent_cli()


if __name__ == "__main__":
    raise SystemExit(main())
