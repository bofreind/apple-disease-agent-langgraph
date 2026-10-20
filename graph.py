# -*- coding: utf-8 -*-
"""LangGraph 图构建 + PostgreSQL 对话状态持久化。

图的拓扑是一个标准 ReAct 循环：

        ┌─────────┐   有 tool_calls    ┌─────────┐
        │  agent  │ ─────────────────▶ │  tools  │
        └─────────┘                    └─────────┘
             ▲                              │
             └──────────── 结果回填 ─────────┘
             │
        （无 tool_calls）
             ▼
            END

- `agent` 节点：调用绑定工具的 LLM，决定"直接回答"还是"调用工具"；
- `tools` 节点：langgraph.prebuilt.ToolNode，自动执行工具并把结果作为 ToolMessage 写回；
- `tools_condition`：条件边，根据最后一条消息里是否含 tool_calls 决定走向 tools 还是 END。

持久化：编译图时传入 checkpointer（PostgresSaver），
LangGraph 会在**每个节点执行后**把整份 AgentState 序列化写入 PostgreSQL，
`thread_id` 即"会话 ID"，同一 thread_id 的多轮调用会自动恢复历史状态。
"""
from __future__ import annotations

import logging
from typing import Optional

import config

logger = logging.getLogger("apple-agent.graph")

# ★ 重要：这个 system prompt 必须是**完全静态的常量**，不允许出现任何逐轮变化的占位符。
#   原因：通义千问的 Context Cache 是自动开启的（前缀重复 ≥1024 token 才命中），
#   命中的前提是"重复内容在前、变化内容在后"。
#   system prompt 位于消息序列最前面，一旦它每轮都变，
#   最长公共前缀会从第一个 token 就分叉，后面整段历史都无法复用，缓存全部失效。
#   因此当前会话的图片/识别状态改由 `_build_live_context()` 追加到消息**末尾**。
SYSTEM_PROMPT = """你是一位专业的苹果病虫害防治助手，服务于果园种植户。

你可以调用以下工具：
1. identify_apple_disease —— 识别当前会话图片的病害类别（健康/锈病/黑星病 scab）。
   用户问"这是什么病/健康吗/得了什么病"时使用。
2. analyze_apple_disease —— 分析当前会话图片的病害严重程度（病斑占比、总体病变率）。
   用户问"严重吗/占比多少/病斑多少"时使用。
3. search_apple_knowledge —— 检索苹果病虫害防治知识库（防治期、用药、管理措施）。
   用户问"怎么治/用什么药/如何防治"时使用。
4. get_orchard_weather —— 查询指定地区当前实况及未来几天的气温、降雨概率、累计雨量、湿度。
   用户问"天气怎样/会不会下雨/湿度高不高/这几天能不能打药"时使用。
   调用必须带地名；用户没说地点时先反问是哪里，不要自己猜。

当前会话是否已有图片、以及系统预先分类的结果，由系统在每轮对话**末尾**以
"【实时上下文】"的形式提供，请据此判断能否调用图片类工具。

工作要求：
- 图片类问题优先调用工具获得真实推理结果，不要凭空猜测；
- 涉及防治方法、用药、农事管理的问题，调用 search_apple_knowledge 检索后再回答；
- 结合对话历史理解用户省略的主语（如多轮追问"那怎么办""严重吗"指的都是上一轮提到的那张图和那个病害）；
- 回答专业、具体、通俗，给出可执行的建议；若工具返回"没有图片"，礼貌提示用户先上传叶片照片。
- 涉及天气与打药时机的判断，先用 get_orchard_weather 查实况，再结合 search_apple_knowledge 给出建议。
输出要求（直接影响响应速度，请严格遵守）：
- 简明扼要，默认把回答控制在 200 字以内；仅当用户明确要求"详细/全面/展开讲讲/分点说明"时才展开成长篇；
- 优先用短句和要点式罗列（每行以"- "开头），不要使用 Markdown 表格、标题层级和代码块；
- 不要复述工具返回的原始数值与字段名，直接给结论和可执行的建议。
"""

# 每轮临时注入的动态上下文模板（追加在消息末尾，不写回 state["messages"]）
_LIVE_CONTEXT_TEMPLATE = """【实时上下文｜系统自动注入，非用户发言，无需向用户复述本段】
{body}
"""


# ==================================================================
# LLM
# ==================================================================
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI

        if not config.LLM_API_KEY:
            raise RuntimeError("未配置 QIANWEN_API_KEY / DASHSCOPE_API_KEY")
        _llm = ChatOpenAI(
            model=config.LLM_MODEL_NAME,
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
        )
    return _llm


# ==================================================================
# Checkpointer（PostgreSQL 持久化）
# ==================================================================
_pool = None
_checkpointer = None


def get_checkpointer():
    """创建 PostgreSQL checkpointer，并自动建表。

    - 使用 psycopg_pool.ConnectionPool 做连接池；
    - autocommit=True + prepare_threshold=0 是 langgraph-checkpoint-postgres 的推荐配置；
    - setup() 会幂等地创建 checkpoints / checkpoint_blobs / checkpoint_writes / checkpoint_migrations 四张表。
    """
    global _pool, _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool

    logger.info("连接 PostgreSQL 检查点库: %s", _mask(config.DB_URL))
    try:
        _pool = ConnectionPool(
            conninfo=config.DB_URL,
            min_size=1,
            max_size=10,
            timeout=10,                                    # 连不上时 10s 内快速失败
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=True,
        )
        _checkpointer = PostgresSaver(_pool)
        _checkpointer.setup()          # 建表（幂等）
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"无法连接 PostgreSQL 检查点库({_mask(config.DB_URL)})：{e}\n"
            f"请确认数据库已启动，例如：\n"
            f"  docker compose -f docker-compose.pg.yml up -d\n"
            f"然后执行：python init_db.py"
        ) from e
    logger.info("PostgreSQL checkpointer 就绪（检查点表已创建）")
    return _checkpointer


def _mask(url: str) -> str:
    """隐藏连接串中的密码，便于打日志"""
    try:
        import re

        return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)
    except Exception:  # noqa: BLE001
        return url


# ==================================================================
# 图构建
# ==================================================================
_graph = None


def _build_live_context(state) -> str:
    """拼装每轮会变化的动态上下文：本会话是否已有图片 + 预分类结果。

    这些内容**不能**放进 system prompt：它们逐轮变化，会让前缀缓存从
    第一个 token 就分叉。改为追加在消息序列末尾后，
    system prompt + 历史消息 构成的最长公共前缀在每轮之间保持稳定，
    Context Cache 才有机会命中（命中部分按输入价 20% 计费且免重复计算）。

    注意：这段内容只用于**本次** LLM 调用，不写回 state，
    因此不会污染持久化到 PostgreSQL 的对话历史。
    """
    parts = []

    image_path = state.get("image_path")
    if image_path:
        parts.append(f"- 当前会话已上传叶片图片（路径 {image_path}），可以调用图片类工具分析。")
    else:
        parts.append("- 当前会话还没有上传图片，图片类工具会返回“没有图片”。")

    cls = state.get("classification") or {}
    if cls.get("success") and cls.get("class") not in (None, "unknown"):
        parts.append(
            f"- 系统已预先分类：{cls.get('class')}（置信度 {cls.get('confidence')}）；"
            f"仅供参考，结论应以工具调用结果为准。"
        )
    else:
        parts.append("- 系统尚未得出有效的预分类结果。")

    return _LIVE_CONTEXT_TEMPLATE.format(body="\n".join(parts))


def build_graph(checkpointer):
    """构建并编译 LangGraph 图（传入 checkpointer 即开启持久化）"""
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    from state import AgentState
    from tools import ALL_TOOLS

    llm_with_tools = get_llm().bind_tools(ALL_TOOLS)

    def agent_node(state: AgentState):
        """LLM 决策节点：直接回答 or 发起工具调用

        消息序列固定为三步：
            [ 静态 system prompt ] + [ 持久化的历史消息 ] + [ 本轮实时上下文 ]
        前两段是稳定前缀（可被 Context Cache 复用），
        只有末尾这一小段逐轮变化，因此缓存不会被破坏。

        `image_path` / `classification` 这类跨轮状态由 LangGraph 从
        PostgreSQL 恢复，所以"第一轮传图、第二轮只问严重吗"时，
        末尾上下文仍会告知模型图片还在。
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        messages.append(HumanMessage(content=_build_live_context(state)))

        response = llm_with_tools.invoke(messages)
        # 只回写 AI 回复；实时上下文不进历史，避免每轮累积冗余消息
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(ALL_TOOLS))

    builder.add_edge(START, "agent")
    # 条件边：最后一条 AIMessage 有 tool_calls -> tools，否则 -> END
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")     # 工具执行完回到 agent 继续推理

    return builder.compile(checkpointer=checkpointer)


def get_graph(force_rebuild: bool = False):
    """全局单例图（编译一次，复用连接池与模型）"""
    global _graph
    if _graph is None or force_rebuild:
        _graph = build_graph(get_checkpointer())
    return _graph


def close_resources() -> None:
    """显式关闭 PostgreSQL 连接池并清空全局单例。

    进程退出前调用，避免连接池 worker 线程残留导致的
    "couldn't stop thread ... within 5.0 seconds" 告警。
    """
    global _pool, _checkpointer, _graph
    if _pool is not None:
        try:
            _pool.close()
            logger.info("PostgreSQL 连接池已关闭")
        except Exception:  # noqa: BLE001
            logger.warning("关闭 PostgreSQL 连接池时出现问题", exc_info=True)
    _pool = None
    _checkpointer = None
    _graph = None
