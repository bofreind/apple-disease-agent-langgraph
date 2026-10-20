# -*- coding: utf-8 -*-
"""工具层：把 Apple/ 已有的三项能力封装成 LangChain Tool，供 LangGraph 智能体调用。

三个工具与 Apple 目录的对应关系：
1. identify_apple_disease → Apple/apple_pest_classify_api.py（ResNet50 分类）
2. analyze_apple_disease  → Apple/seg_api.py + Apple/predict.py（U-Net 分割）
3. search_apple_knowledge → Apple/apple_agent.ipynb（Milvus RAG 检索）

其中两个图片工具**不接收图片路径参数**，而是通过 `InjectedState` 从对话状态里
读取 `image_path`。这样做的意义：
- 图片路径是服务端生成的临时文件，大模型无法凭空知道；
- 状态被持久化到 PostgreSQL 后，用户"第一轮传图、第二轮追问严重程度"也能命中同一张图，
  这正是多轮对话 + 状态持久化的直接收益。
"""
from __future__ import annotations

from itertools import zip_longest
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

import config
import knowledge
import services
from state import AgentState


def _no_image_hint() -> str:
    return ("当前会话中还没有可用的图片。请先上传一张苹果叶片照片（第一轮上传后，"
            "后续追问无需重复上传）。")


@tool
def identify_apple_disease(state: Annotated[AgentState, InjectedState]) -> str:
    """识别当前会话图片中苹果叶片的病害类别（健康 healthy / 锈病 rust / 黑星病 Scab）。
    当用户询问"这是什么病 / 属于哪类病害 / 健不健康 / 得了什么病"时调用。
    无需传参：图片取自当前会话已上传的图片。
    """
    image_path = (state or {}).get("image_path")
    if not image_path:
        return _no_image_hint()

    result = services.classify_image(image_path)
    if not result.get("success"):
        return f"病害识别失败：{result.get('error', '未知错误')}"

    if result.get("class") == "unknown":
        return f"识别结果不确定：{result.get('message', '置信度不足')}"

    zh = {"healthy": "健康", "rust": "锈病", "Scab": "黑星病", "scab": "黑星病"}
    label = zh.get(result["class"], result["class"])
    probs = "，".join(f"{k}:{v:.2%}" for k, v in result.get("all_probs", {}).items())
    return (f"识别结果：{label}（{result['class']}），"
            f"置信度 {result['confidence']:.2%}；各类别概率：{probs}")


@tool
def analyze_apple_disease(state: Annotated[AgentState, InjectedState]) -> str:
    """分析当前会话图片中苹果叶片的病害严重程度（锈病/黑星病病斑占比、总体病变率）。
    当用户询问"严重吗 / 病斑多少 / 占比多少 / 面积多大 / 病情轻重"时调用。
    无需传参：图片取自当前会话已上传的图片。
    """
    image_path = (state or {}).get("image_path")
    if not image_path:
        return _no_image_hint()

    result = services.segment_image(image_path)
    if not result.get("success"):
        return f"分割分析失败：{result.get('error', '未知错误')}"
    if not result.get("has_leaf"):
        return "未检测到叶片区域，请换一张更清晰、叶片占主体的照片。"

    return (
        f"像素级分割结果：叶片占比 {result['leaf_ratio']:.2%}，"
        f"锈病占比 {result['rust_ratio']:.2%}，"
        f"黑星病占比 {result['scab_ratio']:.2%}，"
        f"总体病变率 {result['disease_ratio']:.2%}"
        + ("，叶片已感染病害。" if result.get("has_disease") else "，未检测到明显病斑。")
    )


@tool
def search_apple_knowledge(query: str) -> str:
    """检索苹果病虫害防治知识库（来源：Apple/apple.txt）。
    当用户询问防治方法、用药、关键防治期、病害/虫害识别与管理等农业知识时调用。
    参数:
        query: 检索关键词或问题，例如"锈病怎么防治""幼果期用药"。
    """
    hits = knowledge.retrieve(query, limit=config.RETRIEVE_LIMIT)
    if not hits:
        return "知识库未检索到相关内容（可能 Milvus 未启动或知识库为空）。"

    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] 来源：{h['source']}（相关度 {h['score']}）\n{h['text']}")
    return "\n\n".join(lines)

# Open-Meteo 返回的 weather_code 遵循 WMO 标准，翻译成模型（和用户）能直接读懂的中文
_WMO_CODE = {
    0: "晴", 1: "晴间多云", 2: "多云", 3: "阴",
    45: "雾", 48: "雾凇",
    51: "弱毛毛雨", 53: "毛毛雨", 55: "强毛毛雨",
    56: "冻毛毛雨", 57: "强冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "米雪",
    80: "阵雨", 81: "强阵雨", 82: "暴雨",
    85: "阵雪", 86: "强阵雪",
    95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "雷阵雨伴强冰雹",
}

# 降雨概率/雨量的农业判读阈值，拼在工具返回值末尾，引导模型做出可执行的打药建议
_WEATHER_HINT = (
    "判读口径：降雨概率≥60%或累计雨量>2mm 不宜喷施（药液易被冲刷）；"
    "雨后连续高湿（≥85%）是病菌侵染窗口，宜用保护性杀菌剂。"
)


@tool
def get_orchard_weather(city: str) -> str:
    """查询指定地区当前实况及未来几天的气温、降雨概率、累计雨量和湿度，
    用于判断打药时机与病害侵染风险。
    当用户询问某地的天气、是否下雨、能不能打药、近期湿度如何时调用。
    参数:
        city: 中文地名，如"烟台市""延安"。用户没说明地点时不要调用，先反问是哪里。
    """
    r = services.get_weather(city)
    if not r.get("success"):
        return f"天气查询失败：{r.get('error')}，请换个更明确的地名。"

    raw = r["raw"]
    cur = raw.get("current") or {}

    lines = [
        f"{r['place']}当前实况："
        f"气温 {cur.get('temperature_2m')}℃，"
        f"湿度 {cur.get('relative_humidity_2m')}%，"
        f"降水 {cur.get('precipitation')}mm，"
        f"{_WMO_CODE.get(cur.get('weather_code'), '天气状况未知')}。"
    ]

    # 逐日预报：services 层已把 daily 拉回来了，必须原样交给模型，
    # 否则模型只能看到实况，无法回答"明天/这几天能不能打药"。
    daily = raw.get("daily") or {}
    days = daily.get("time") or []
    if days:
        lines.append("未来几天预报：")
        cols = (
            daily.get("temperature_2m_min") or [],
            daily.get("temperature_2m_max") or [],
            daily.get("precipitation_probability_max") or [],
            daily.get("precipitation_sum") or [],
        )
        for d, tmin, tmax, prob, psum in zip_longest(days, *cols, fillvalue="未知"):
            lines.append(
                f"- {d}：气温 {tmin}~{tmax}℃，"
                f"降雨概率 {prob}%，累计雨量 {psum}mm"
            )

    lines.append(_WEATHER_HINT)
    return "\n".join(lines)

# 供 LangGraph 绑定的工具列表
ALL_TOOLS = [identify_apple_disease, analyze_apple_disease,
             search_apple_knowledge, get_orchard_weather]
