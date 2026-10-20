# -*- coding: utf-8 -*-
"""持久化端到端验证脚本（需要 PostgreSQL 已启动）。

它会真实验证三件事：
  A. 多轮对话：同一 thread_id 第二轮不带图片，仍能复用第一轮的 image_path；
  B. 进程级持久化：**新建** graph/连接池（模拟进程重启）后，用同一 thread_id
     仍能读回完整历史 —— 证明状态来自 PostgreSQL 而不是内存；
  C. 落库情况：直接查 checkpoints / checkpoint_writes 表打印检查点数量。

运行：
    python init_db.py           # 先建表
    python verify_persistence.py
"""
import sys

sys.path.insert(0, ".")

from langgraph.checkpoint.postgres import PostgresSaver  # noqa: E402
from psycopg_pool import ConnectionPool  # noqa: E402

import config  # noqa: E402
import graph as graph_mod  # noqa: E402
import services  # noqa: E402


def step(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def ask(app, tid: str, text: str, **extra):
    """向图发一轮消息（只传增量）"""
    inputs = {"messages": [{"role": "user", "content": text}], **extra}
    result = app.invoke(inputs, config={"configurable": {"thread_id": tid}})
    tools = [tc["name"] for m in result["messages"]
             if getattr(m, "tool_calls", None) for tc in m.tool_calls]
    answer = next((m.content for m in reversed(result["messages"])
                   if m.__class__.__name__ == "AIMessage" and m.content), "")
    return result, tools, answer


def main() -> None:
    import glob

    img = glob.glob("../Apple/test/rust/*.jpg")[0]

    # ---------------- A. 多轮对话 + 跨轮图片复用 ----------------
    step("A. 多轮对话（第 1 轮带图，第 2 轮不带图）")
    app = graph_mod.get_graph()
    tid = "verify-multiturn"
    graph_mod.get_checkpointer().delete_thread(tid)     # 干净开始

    cls = services.classify_image(img)
    r1, t1, a1 = ask(app, tid, "看看这张叶子是什么病？", image_path=img, classification=cls)
    print(f"第1轮 工具={t1}  image_path={r1.get('image_path')}")
    print(f"      答: {a1[:100]}")

    r2, t2, a2 = ask(app, tid, "严重吗？病斑占多少？")     # 注意：第二轮没有传图片
    print(f"第2轮 工具={t2}  image_path={r2.get('image_path')}")
    print(f"      答: {a2[:100]}")
    assert r2.get("image_path") == img, "跨轮 image_path 未保留 ❌"
    assert "analyze_apple_disease" in t2, "第二轮未调用分割工具 ❌"
    print("✅ A 通过：第二轮未传图片，仍复用第一轮的图片并调用了分割工具")

    # ---------------- B. 进程级持久化 ----------------
    step("B. 进程级持久化（新建连接池/图，模拟进程重启）")
    graph_mod._pool.close()
    graph_mod._pool = None
    graph_mod._checkpointer = None
    graph_mod._graph = None                              # 丢弃全部内存对象
    print("已销毁内存中的 graph / checkpointer / pool，重新从 PostgreSQL 构建 ...")

    app2 = graph_mod.get_graph()
    snap = app2.get_state({"configurable": {"thread_id": tid}})
    msgs = (snap.values or {}).get("messages", [])
    print(f"恢复后消息数={len(msgs)}  image_path={snap.values.get('image_path')}")
    for m in msgs:
        print(f"   [{m.__class__.__name__}] {str(m.content)[:70]}")
    assert len(msgs) >= 4, "历史未从 PostgreSQL 恢复 ❌"
    print("✅ B 通过：新进程用同一 thread_id 恢复了完整历史")

    # 继续第 3 轮，验证可续写
    r3, t3, a3 = ask(app2, tid, "那这个病该怎么防治？")
    print(f"第3轮 工具={t3}")
    print(f"      答: {a3[:100]}")
    print("✅ 续写成功")

    # ---------------- C. 直接查库 ----------------
    step("C. PostgreSQL 落库情况")
    pool = graph_mod._pool
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT thread_id, COUNT(*) AS ckpts
            FROM checkpoints GROUP BY thread_id ORDER BY thread_id
            """
        ).fetchall()
        print("thread_id -> 检查点数：")
        for r in rows:
            print(f"   {r[0]:<24} {r[1]}")
        n_writes = conn.execute("SELECT COUNT(*) FROM checkpoint_writes").fetchone()[0]
        n_blobs = conn.execute("SELECT COUNT(*) FROM checkpoint_blobs").fetchone()[0]
        print(f"checkpoint_writes={n_writes}  checkpoint_blobs={n_blobs}")
    print("✅ C 通过")

    print("\n🎉 持久化端到端验证全部通过")


if __name__ == "__main__":
    try:
        main()
    finally:
        graph_mod.close_resources()      # 显式关闭连接池，避免退出告警
