"""
main.py - 项目入口
运行方式：python main.py
"""

from infra import configure_logging

configure_logging()

from data_pipeline.pipeline import run

if __name__ == "__main__":
    run()
