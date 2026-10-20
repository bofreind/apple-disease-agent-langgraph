# -*- coding: utf-8 -*-
"""命令行多轮对话 Demo：验证"跨轮状态 + PostgreSQL 持久化"。

用法：
    python demo_cli.py                        # 新会话
    python demo_cli.py <thread_id>            # 继续某个已存在的会话（历史自动恢复）
    python demo_cli.py <thread_id> --image Apple_test/xx.jpg   # 首轮带图片

交互指令：/history 查看历史，/exit 退出。
"""
import sys
from typing import Any, Optional

import graph as graph_mod


def _thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _extract_answer(result: dict) -> str:
    for m in reversed(result.get("messages", [])):
        if m.__class__.__name__ == "AIMessage" and getattr(m, "content", ""):
            return m.content
    return "(无回答)"


def run(thread_id: str, image_path: Optional[str] = None) -> None:
    graph = graph_mod.get_graph()
    print(f"会话 thread_id = {thread_id}（状态持久化在 PostgreSQL）")
    print("输入问题开始对话，/history 查看历史，/exit 退出\n")

    pending_image = image_path          # 只在第一轮附加图片

    while True:
        try:
            question = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question == "/exit":
            break
        if question == "/history":
            snap = graph.get_state(_thread_config(thread_id))
            for m in (snap.values or {}).get("messages", []):
                print(f"  [{m.__class__.__name__}] {str(m.content)[:120]}")
            continue

        inputs: dict[str, Any] = {
            "messages": [{"role": "user", "content": question}],
        }
        # 首轮且有图片：写入 image_path（此后该值会被 checkpointer 记住，跨轮可用）
        if pending_image:
            import services

            inputs["image_path"] = pending_image
            cls = services.classify_image(pending_image)
            inputs["classification"] = cls
            print(f"（已上传图片并预分类：{cls.get('class')}）")
            pending_image = None

        result = graph.invoke(inputs, config=_thread_config(thread_id))
        print(f"助手> {_extract_answer(result)}\n")


if __name__ == "__main__":
    tid = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "demo-thread"
    img = None
    if "--image" in sys.argv:
        img = sys.argv[sys.argv.index("--image") + 1]
    run(tid, img)
