# -*- coding: utf-8 -*-
"""自检脚本：验证各组件是否就绪（PostgreSQL / Milvus / 模型 / 图）。

运行：
    python selftest.py
"""
import sys

sys.path.insert(0, ".")

OK = "[OK]  "
BAD = "[FAIL]"


def check(name, fn):
    try:
        detail = fn()
        print(f"{OK}{name}" + (f"  -> {detail}" if detail else ""))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"{BAD}{name}  -> {type(e).__name__}: {str(e)[:150]}")
        return False


def main() -> None:
    import config

    print("=" * 60)
    print("langgraphApple 自检")
    print("=" * 60)

    results = []
    results.append(check("配置加载", lambda: f"DB={graph_mask()} | LLM={config.LLM_MODEL_NAME} | 类别={config.CLASS_NAMES}"))

    results.append(check("PostgreSQL 检查点库", lambda: __import__("graph").get_checkpointer() and "连接成功，检查点表就绪"))

    def _cls():
        import services

        r = services.classify_image(__import__("glob").glob("../Apple/test/rust/*.jpg")[0])
        return f"分类可用：{r.get('class')} {r.get('confidence')}" if r.get("success") else (_ for _ in ()).throw(RuntimeError(r.get("error")))

    results.append(check("分类模型（复用 Apple ResNet50）", _cls))

    def _seg():
        import services

        r = services.segment_image(__import__("glob").glob("../Apple/test/rust/*.jpg")[0])
        return f"分割可用：病变率 {r.get('disease_ratio')}" if r.get("success") else (_ for _ in ()).throw(RuntimeError(r.get("error")))

    results.append(check("分割模型（复用 Apple U-Net）", _seg))

    results.append(check("Milvus 知识库", lambda: (lambda c: "连接成功" if c else (_ for _ in ()).throw(RuntimeError("未连接（请先启动 Milvus 并执行 ingest_knowledge.py）")))(__import__("knowledge").get_milvus_client())))

    def _graph():
        import graph as g

        app = g.get_graph()
        return f"节点={list(app.get_graph().nodes)}"

    results.append(check("LangGraph 图编译", _graph))

    print("-" * 60)
    print(f"通过 {sum(results)}/{len(results)}")
    print("=" * 60)


def graph_mask() -> str:
    import graph as g

    import config

    return g._mask(config.DB_URL)


if __name__ == "__main__":
    try:
        main()
    finally:
        import graph as _g

        _g.close_resources()          # 显式关闭连接池，避免退出告警
