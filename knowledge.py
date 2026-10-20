# -*- coding: utf-8 -*-
"""RAG 知识库检索层：**复用 Apple/apple_agent.ipynb 的 Milvus + BGE-M3 方案**。

复用点：
- Milvus 连接 / 建库 / 建集合（rag_tutorial / Appledocs / COSINE / 1024 维）
- 嵌入模型 DashScopeEmbeddings(Pro/BAAI/bge-m3)
- 知识来源 Apple/apple.txt（苹果病虫害防治）
- 检索结果字段 chunk_id / text / source，并按 COSINE 距离做相关度过滤

所有资源懒加载，Milvus 未启动时不影响"纯文本问答"，只是检索工具会返回提示。
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

import config

logger = logging.getLogger("apple-agent.knowledge")

_lock = threading.Lock()
_milvus_client = None
_embedding_model = None


def get_embedding_model():
    """嵌入模型（单例）。复用 notebook 中的 DashScopeEmbeddings + bge-m3。"""
    global _embedding_model
    if _embedding_model is None:
        with _lock:
            if _embedding_model is None:
                from langchain_community.embeddings import DashScopeEmbeddings

                _embedding_model = DashScopeEmbeddings(model=config.EMBED_MODEL_NAME)
    return _embedding_model


def get_milvus_client():
    """Milvus 客户端（单例）。连不上返回 None，调用方需处理。"""
    global _milvus_client
    if _milvus_client is not None:
        return _milvus_client
    with _lock:
        if _milvus_client is not None:
            return _milvus_client
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(config.MILVUS_URI, timeout=5)
            db_names = client.list_databases()
            if config.MILVUS_DB_NAME not in db_names:
                client.create_database(db_name=config.MILVUS_DB_NAME)
            client.use_database(db_name=config.MILVUS_DB_NAME)
            _milvus_client = client
            logger.info("Milvus 连接成功: %s/%s", config.MILVUS_URI, config.COLLECTION_NAME)
        except Exception as e:  # noqa: BLE001
            logger.warning("Milvus 连接失败: %s（知识库检索暂不可用）", e)
            _milvus_client = None
    return _milvus_client


def ensure_collection(client) -> None:
    """确保 Appledocs 集合存在（复用 notebook 中的 dimension/metric_type 配置）"""
    if not client.has_collection(collection_name=config.COLLECTION_NAME):
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            dimension=config.EMBED_DIM,
            metric_type="COSINE",
        )
        logger.info("已创建集合: %s", config.COLLECTION_NAME)


def _collection_auto_id(client) -> bool:
    """判断集合主键是否 auto_id。

    quick-mode 建出来的集合 auto_id=True（主键自增，insert 不需要 id）；
    而用显式 schema 建的集合 auto_id=False（insert 必须提供 id）。
    两种情况都要兼容，否则会报 "Insert missed an field `id`"。
    """
    try:
        return bool(client.describe_collection(config.COLLECTION_NAME).get("auto_id"))
    except Exception:  # noqa: BLE001
        return False


def retrieve(query: str, limit: Optional[int] = None) -> list[dict[str, Any]]:
    """向量检索，返回 [{text, source, score}]（按相关度过滤，score 越小越相关）"""
    client = get_milvus_client()
    if client is None:
        return []
    limit = limit or config.RETRIEVE_LIMIT

    query_vector = get_embedding_model().embed_query(query)
    results = client.search(
        collection_name=config.COLLECTION_NAME,
        data=[query_vector],
        limit=max(limit * 2, 5),
        output_fields=["chunk_id", "text", "source"],
        timeout=10,
    )

    hits: list[dict[str, Any]] = []
    for item in results[0]:
        distance = float(item.get("distance", 1.0))
        if distance >= config.RETRIEVE_SCORE_THRESHOLD:   # COSINE 距离过滤
            continue
        entity = item.get("entity", item)
        hits.append({
            "text": entity.get("text", ""),
            "source": entity.get("source", "apple.txt"),
            "score": round(distance, 4),
        })
    return hits[:limit]


# ------------------------------------------------------------------
# 知识入库（离线脚本 ingest_knowledge.py 使用）
# ------------------------------------------------------------------
def _split_text(text: str, max_len: int = 400) -> list[str]:
    """按空行切段，再把过短段落合并到 max_len 左右"""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_len:
            buf = f"{buf}\n{p}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def ingest_knowledge_file(path: Optional[str] = None) -> int:
    """读取 apple.txt -> 切分 -> 向量化 -> 写入 Milvus，返回写入条数"""
    path = path or config.KNOWLEDGE_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(f"知识库文件不存在: {path}")

    with open(path, encoding="utf-8") as f:
        raw = f.read()

    chunks = _split_text(raw)
    if not chunks:
        return 0

    client = get_milvus_client()
    if client is None:
        raise RuntimeError("Milvus 未连接，无法写入知识库")
    ensure_collection(client)

    # 清空后重建，保证幂等
    try:
        client.delete(collection_name=config.COLLECTION_NAME, filter="id >= 0")
    except Exception:  # noqa: BLE001
        pass

    embed = get_embedding_model()
    vectors = embed.embed_documents(chunks)
    source = os.path.basename(path)

    auto_id = _collection_auto_id(client)
    data: list[dict[str, Any]] = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        row: dict[str, Any] = {"vector": vec, "chunk_id": i, "text": chunk, "source": source}
        if not auto_id:            # 显式 schema 的集合必须自带主键
            row["id"] = i
        data.append(row)

    client.insert(collection_name=config.COLLECTION_NAME, data=data)
    client.flush(collection_name=config.COLLECTION_NAME)
    logger.info("知识库写入完成: %d 条", len(data))
    return len(data)
