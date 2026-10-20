# -*- coding: utf-8 -*-
"""对话状态定义（LangGraph State）。

LangGraph 的"状态"就是一个带 reducer 的 TypedDict：
- `messages` 使用内置的 `add_messages` reducer —— 新消息会"追加/合并"到历史里，
  而不是覆盖，这是多轮对话能记住上下文的基础；
- 其余字段（image_path / classification ...）没有配 reducer，
  语义是"最后一次写入生效"，且**本轮不传该字段时保持上一轮的值不变**
  （这正是"用户第一轮传图、后续轮次追问"能复用图片的关键）。
"""
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # ---- 对话消息（带 add_messages reducer，自动累积多轮历史） ----
    messages: Annotated[list, add_messages]

    # ---- 会话 / 用户标识（便于按用户维度做多会话隔离） ----
    thread_id: Optional[str]
    user_id: Optional[str]

    # ---- 当前会话关联的图片路径 ----
    # 用户上传一次图片后写入，后续轮次即使不再上传也能被工具复用（跨轮状态）
    image_path: Optional[str]

    # ---- 缓存的本轮分类结果（供回答与接口返回复用，避免重复推理） ----
    classification: Optional[dict[str, Any]]

    # ---- 本轮命中的知识库片段 ----
    retrieved: Optional[list[dict[str, Any]]]
