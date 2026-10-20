# -*- coding: utf-8 -*-
"""把 Apple/apple.txt 知识库写入 Milvus（复用 apple_agent.ipynb 的流程）。

前置条件：Milvus 已启动（默认 http://localhost:19530）。

运行：
    python ingest_knowledge.py
"""
import logging

from knowledge import get_milvus_client, ingest_knowledge_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest")


def main() -> None:
    client = get_milvus_client()
    if client is None:
        logger.error("Milvus 未连接，请先启动 Milvus（docker compose up -d）")
        return

    n = ingest_knowledge_file()
    logger.info("知识库写入完成，共 %d 条", n)

    stats = client.get_collection_stats("Appledocs")
    logger.info("集合 %s 当前数据量: %s", "Appledocs", stats.get("row_count", 0))


if __name__ == "__main__":
    main()
