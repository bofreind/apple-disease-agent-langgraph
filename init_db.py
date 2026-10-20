# -*- coding: utf-8 -*-
"""初始化 PostgreSQL 中的 LangGraph 检查点表。

`PostgresSaver.setup()` 会幂等创建：
    - checkpoints            每个 super-step 的状态快照
    - checkpoint_blobs       状态中较大的二进制字段（如消息内容）
    - checkpoint_writes      节点写入记录（用于恢复/重放）
    - checkpoint_migrations  迁移版本表

运行：
    python init_db.py
"""
import logging

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

import config
import graph as graph_mod

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("init_db")


def main() -> None:
    logger.info("目标数据库: %s", graph_mod._mask(config.DB_URL))

    with ConnectionPool(
        conninfo=config.DB_URL,
        max_size=4,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=True,
    ) as pool:
        saver = PostgresSaver(pool)
        saver.setup()          # 幂等建表
        logger.info("检查点表已创建/校验完成")

        with pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name LIKE 'checkpoint%%'
                ORDER BY table_name
                """
            ).fetchall()
        logger.info("已存在检查点相关表: %s", [r[0] for r in rows])

    logger.info("初始化完成 ✅")


if __name__ == "__main__":
    main()
