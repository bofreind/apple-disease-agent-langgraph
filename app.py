# -*- coding: utf-8 -*-
"""苹果病虫害问答智能体 —— FastAPI 服务（带多轮对话 + PostgreSQL 持久化）。

接口：
- POST /chat                 多轮问答（同一 thread_id 即同一会话，历史自动从 PostgreSQL 恢复）
- POST /chat/stream          同上，但以 SSE 流式返回（首字即出，无需等待全文生成）
- GET  /history/{thread_id}  查看某会话的完整消息历史
- GET  /threads              列出所有会话（含最后更新时间近似值）
- GET  /health               健康检查

实现要点：
- `/chat` 为同步 def，FastAPI 会自动放到线程池执行，避免阻塞事件循环。
- `/chat/stream` 为异步 def：图本身是同步的，因此放在**后台线程**里跑，
  通过 asyncio 队列把流式分片交回事件循环，保证事件循环不被阻塞。
- 图片以 base64 传入，服务端落盘为临时文件，路径写入对话状态 image_path；
  因为状态被持久化，后续轮次不传图片也能继续分析同一张图。

`/chat/stream` 的 SSE 事件协议（每条都是 `data: {json}` + 空行）：

| type | 载荷 | 含义 |
|---|---|---|
| `start` | `thread_id` | 会话已确定，可立即回写前端 |
| `meta`  | `class / class_index / confidence / all_probs` | 识别结果（本轮算的或历史回退的），前端可立刻点亮卡片 |
| `status`| `text` | 工具执行进度提示（如"正在分析病斑严重程度…"） |
| `token` | `text` | 回答的一个增量分片 |
| `reset` | — | 丢弃此前已显示的分片（决策轮的短文本不应展示） |
| `done`  | `thread_id / answer / class…` | 结束；`answer` 为持久化状态里的权威全文 |
| `error` | `message` | 出错 |
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import config
import graph as graph_mod
import services

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("apple-agent")


# ==================================================================
# 请求 / 响应模型
# ==================================================================
class ChatRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None      # 不传则自动生成（新会话）
    user_id: Optional[str] = None        # 预留：按用户维度隔离多会话
    image_base64: Optional[str] = None   # 可选：本轮上传的叶片图片


# ==================================================================
# 生命周期
# ==================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("正在初始化：PostgreSQL 检查点表 ...")
    graph_mod.get_checkpointer()         # 建表 + 校验连接

    # 预热：把「导入 torch/图编译/加载两个模型/连 Milvus」的一次性开销
    # 从"首位用户的首个请求"挪到服务启动阶段。
    # 不做这一步时，第一个请求会把 ≈15s 的冷启动算在用户头上（流式下首字能到 18s）。
    logger.info("预热中：编译图 + 加载分类/分割模型 + 连接 Milvus ...")

    def _warm_up() -> None:
        graph_mod.get_graph()            # 触发 tools/services 导入与图编译
        services._load_classifier()
        services._load_segmenter()
        try:
            import knowledge

            knowledge.get_milvus_client()
            knowledge.get_embedding_model()
        except Exception as e:  # noqa: BLE001
            logger.warning("Milvus 预热失败（不影响启动，工具层会自行降级）: %s", e)

    try:
        await asyncio.to_thread(_warm_up)
        logger.info("预热完成")
    except Exception as e:  # noqa: BLE001
        logger.warning("预热失败（不影响服务启动）: %s", e)

    logger.info("初始化完成，服务就绪")
    yield
    logger.info("服务关闭")


app = FastAPI(title="苹果病虫害问答智能体（LangGraph）", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================================
# 工具函数
# ==================================================================
def _save_image(image_base64: str) -> Optional[str]:
    """把 base64 图片落盘为临时文件，返回路径"""
    try:
        raw = base64.b64decode(image_base64)
        fd, path = tempfile.mkstemp(prefix="apple_", suffix=".jpg")
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        return path
    except Exception as e:  # noqa: BLE001
        logger.warning("图片保存失败: %s", e)
        return None


def _thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _build_turn(request: ChatRequest):
    """组装本轮输入：落盘图片 + 预分类。

    只写入发生变化的字段（`image_path` / `classification`），其余字段沿用
    checkpoint 里的历史值 —— AgentState 里除 messages 外都没有 reducer，
    显式传 None 会**覆盖**掉历史值。

    返回 (thread_id, config, inputs, class_result)。
    """
    from langchain_core.messages import HumanMessage

    thread_id = request.thread_id or str(uuid.uuid4())
    inputs: dict[str, Any] = {
        "messages": [HumanMessage(content=(request.question or "").strip())],
        "thread_id": thread_id,
        "user_id": request.user_id,
    }

    class_result: Optional[dict] = None
    if request.image_base64:
        image_path = _save_image(request.image_base64)
        if image_path:
            inputs["image_path"] = image_path          # 写入状态（并随 checkpointer 持久化）
            class_result = services.classify_image(image_path)
            inputs["classification"] = class_result    # 缓存分类结果，避免重复推理
    return thread_id, _thread_config(thread_id), inputs, class_result


def _class_fields(class_result: Optional[dict]) -> dict:
    """把分类结果摊平成响应字段（识别失败时返回空字典）"""
    if class_result and class_result.get("success"):
        return {
            "class": class_result.get("class"),
            "class_index": class_result.get("class_index"),
            "confidence": class_result.get("confidence"),
            "all_probs": class_result.get("all_probs", {}),
        }
    return {}


def _sse(payload: dict) -> str:
    """打包一条 SSE 事件（ensure_ascii=False 保证中文可读）"""
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _chunk_text(msg) -> str:
    """取出流式分片的文本（兼容 content 为 str 或分块 list 两种情况）"""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


# 工具名 → 前端可读的进度提示
TOOL_STATUS = {
    "identify_apple_disease": "正在识别叶片病害类别…",
    "analyze_apple_disease": "正在分析病斑严重程度…",
    "search_apple_knowledge": "正在检索防治知识库…",
    "get_orchard_weather": "正在查询果园天气…",
}


# ==================================================================
# 接口
# ==================================================================
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "db_url": graph_mod._mask(config.DB_URL),
        "threads": _count_threads(),
        "milvus_ok": _milvus_ok(),
    }


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    """多轮问答主接口"""
    question = (request.question or "").strip()
    if not question:
        return {"status": "error", "message": "question 不能为空"}

    # ---- 组装本轮输入（只写入发生变化的字段，其余保持历史值） ----
    thread_id, config_, inputs, class_result = _build_turn(request)

    # ---- 调用图（checkpointer 会自动加载该 thread 的历史状态） ----
    try:
        result = graph_mod.get_graph().invoke(inputs, config=config_)
    except Exception as e:  # noqa: BLE001
        logger.exception("图执行异常")
        return {"status": "error", "message": f"智能体执行失败: {e}", "thread_id": thread_id}

    # ---- 本轮没传图时，回退到状态里持久化的分类结果 ----
    # classification 写在 LangGraph 状态中（随 checkpoint 落 PostgreSQL），
    # 所以第 2 轮起即使不再传图，也能把本会话的识别结论回传给前端。
    if class_result is None:
        cached = result.get("classification")
        if isinstance(cached, dict) and cached.get("success"):
            class_result = cached

    answer = ""
    for msg in reversed(result.get("messages", [])):
        if msg.__class__.__name__ == "AIMessage" and getattr(msg, "content", ""):
            answer = msg.content
            break

    response: dict[str, Any] = {
        "status": "success",
        "thread_id": thread_id,
        "answer": answer,
        "metadata": {"has_image": bool(request.image_base64 or result.get("image_path"))},
    }
    if class_result:
        response.update(_class_fields(class_result))
    return response


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """多轮问答（SSE 流式版）。

    与非流式 `/chat` 的唯一区别是返回形式：识别结果先出（`meta` 事件），
    回答正文逐 token 推送，用户不必等全文生成完（实测首字 ≈0.7s，而全文要 6s 左右）。
    历史、持久化、多轮语义完全一致 —— 两者共用 `_build_turn` 与同一张图。
    """
    if not (request.question or "").strip():
        return StreamingResponse(_single_event({"type": "error", "message": "question 不能为空"}),
                                 media_type="text/event-stream")
    return StreamingResponse(
        _event_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",      # 让 nginx 类反代不要缓冲
        },
    )


async def _single_event(payload: dict) -> AsyncIterator[str]:
    yield _sse(payload)


async def _event_stream(request: ChatRequest) -> AsyncIterator[str]:
    """SSE 事件流生产者。

    图是同步实现（内部有 torch 推理与同步 LLM 调用），因此放到后台线程执行，
    用 asyncio 队列把分片交回事件循环 —— 这样既保住了同步代码，
    又不会阻塞事件循环导致其他请求排队。
    """
    # ---- 1. 预处理：落盘图片 + 预分类（放线程里做，避免阻塞事件循环） ----
    try:
        thread_id, config_, inputs, class_result = await asyncio.to_thread(_build_turn, request)
    except Exception as e:  # noqa: BLE001
        logger.exception("预处理失败")
        yield _sse({"type": "error", "message": f"预处理失败: {e}"})
        return

    yield _sse({"type": "start", "thread_id": thread_id})

    # ---- 2. 识别结果：本轮没传图就回退到状态里持久化的那份 ----
    if class_result is None:
        try:
            snapshot = await asyncio.to_thread(
                lambda: graph_mod.get_graph().get_state(config_)
            )
            cached = (snapshot.values or {}).get("classification")
            if isinstance(cached, dict) and cached.get("success"):
                class_result = cached
        except Exception as e:  # noqa: BLE001
            logger.warning("读取历史分类结果失败: %s", e)

    yield _sse({"type": "meta", "thread_id": thread_id, **_class_fields(class_result)})

    # ---- 3. 后台线程跑图，逐块回传 ----
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    done_marker = object()

    def _worker() -> None:
        try:
            stream = graph_mod.get_graph().stream(
                inputs, config=config_, stream_mode=["messages", "updates"]
            )
            for item in stream:
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(queue.put_nowait, ("__error__", exc))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done_marker)

    threading.Thread(target=_worker, daemon=True, name="graph-stream").start()

    buf: list[str] = []
    while True:
        item = await queue.get()
        if item is done_marker:
            break

        mode, chunk = item
        if mode == "__error__":
            logger.exception("图执行异常", exc_info=chunk)
            yield _sse({"type": "error", "message": f"智能体执行失败: {chunk}"})
            return

        if mode == "messages":
            msg, meta = chunk
            # 只转发 agent 节点产出的分片（工具节点不产生自然语言）
            if (meta or {}).get("langgraph_node") != "agent":
                continue

            # (a) 工具调用决策：把工具名翻成中文进度提示
            for tool_chunk in (getattr(msg, "tool_call_chunks", None) or []):
                name = tool_chunk.get("name")
                if not name:
                    continue
                if buf:
                    # 决策轮若已吐出过短文本（如"我来看看这张图"），不应拼进最终回答
                    buf.clear()
                    yield _sse({"type": "reset"})
                yield _sse({"type": "status",
                            "text": TOOL_STATUS.get(name, f"正在调用 {name}…")})

            # (b) 正文增量
            text = _chunk_text(msg)
            if text:
                buf.append(text)
                yield _sse({"type": "token", "text": text})

        elif mode == "updates":
            # 工具执行完毕 → 提示"正在整理回答"
            if isinstance(chunk, dict) and "tools" in chunk:
                yield _sse({"type": "status", "text": "已获得分析结果，正在整理回答…"})

    # ---- 4. 收尾：以持久化状态里的全文为准，避免个别分片丢失 ----
    answer = "".join(buf).strip()
    try:
        snapshot = await asyncio.to_thread(
            lambda: graph_mod.get_graph().get_state(config_)
        )
        values = snapshot.values or {}
        for msg in reversed(values.get("messages", [])):
            if msg.__class__.__name__ == "AIMessage" and getattr(msg, "content", ""):
                answer = msg.content
                break
        final_cls = values.get("classification")
        if isinstance(final_cls, dict) and final_cls.get("success"):
            class_result = final_cls
    except Exception as e:  # noqa: BLE001
        logger.warning("读取最终状态失败: %s", e)

    yield _sse({"type": "done", "thread_id": thread_id, "answer": answer,
                **_class_fields(class_result)})


@app.get("/history/{thread_id}")
def history(thread_id: str) -> dict[str, Any]:
    """读取某会话的历史消息（直接来自 PostgreSQL 中的 checkpoint 状态）"""
    snapshot = graph_mod.get_graph().get_state(_thread_config(thread_id))
    messages = []
    for m in (snapshot.values or {}).get("messages", []):
        role = {"HumanMessage": "user", "AIMessage": "assistant",
                "ToolMessage": "tool", "SystemMessage": "system"}.get(
            m.__class__.__name__, m.__class__.__name__)
        content = m.content if isinstance(m.content, str) else str(m.content)
        item = {"role": role, "content": content}
        if role == "assistant" and getattr(m, "tool_calls", None):
            item["tool_calls"] = [tc.get("name") for tc in m.tool_calls]
        if role == "tool":
            item["name"] = getattr(m, "name", None)
        messages.append(item)
    return {
        "status": "success",
        "thread_id": thread_id,
        "image_path": (snapshot.values or {}).get("image_path"),
        "messages": messages,
    }


@app.get("/threads")
def threads(limit: int = 50) -> dict[str, Any]:
    """列出全部会话（按最近 checkpoint 倒序）"""
    return {"status": "success", "threads": _list_threads(limit)}


# ==================================================================
# 直接查 PG 的辅助函数（读 checkpoint 表）
# ==================================================================
def _pool():
    graph_mod.get_checkpointer()          # 确保 pool 已建
    return graph_mod._pool


def _list_threads(limit: int = 50) -> list[dict[str, Any]]:
    try:
        with _pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT thread_id, MAX(checkpoint_id) AS last_ckpt, COUNT(*) AS ckpt_count
                FROM checkpoints
                GROUP BY thread_id
                ORDER BY last_ckpt DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return [{"thread_id": r[0], "last_checkpoint": str(r[1]), "checkpoints": r[2]}
                for r in rows]
    except Exception as e:  # noqa: BLE001
        logger.warning("查询会话列表失败: %s", e)
        return []


def _count_threads() -> int:
    try:
        with _pool().connection() as conn:
            row = conn.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints").fetchone()
        return int(row[0]) if row else 0
    except Exception:  # noqa: BLE001
        return 0


def _milvus_ok() -> bool:
    try:
        import knowledge

        return knowledge.get_milvus_client() is not None
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.AGENT_SERVICE_HOST, port=config.AGENT_SERVICE_PORT)
