"""
infra/logging_config.py

全局日志配置：统一格式与级别，各模块仅使用 logging.getLogger(__name__)。
在程序入口（main.py 或脚本 __main__）调用 configure_logging() 一次即可。
若 root 已有 handler（如 eval 脚本先配置），则不再重复添加，避免双份输出。
"""

from __future__ import annotations

import logging


def configure_logging(
    level: int = logging.INFO,
    format_string: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt: str = "%H:%M:%S",
) -> None:
    """
    为 root logger 配置默认 handler 与格式（仅当尚未配置时）。
    应在 main.py 或各脚本的 __main__ 入口最先调用。
    """
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt=datefmt,
    )


# TODO: agent 开发完成后在此增加 FileHandler / RotatingFileHandler