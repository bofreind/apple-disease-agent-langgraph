# -*- coding: utf-8 -*-
"""集中配置：所有路径 / 端口 / 模型 / 数据库连接统一从这里读取。

复用 Apple/ 目录中 "config.py 集中配置" 的组织方式，
在其基础上新增 LangGraph + PostgreSQL 持久化相关配置。
"""
import os

from dotenv import load_dotenv

# ============ 目录 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../langgraphApple
PROJECT_ROOT = os.path.dirname(BASE_DIR)                        # .../python学习
APPLE_DIR = os.path.join(PROJECT_ROOT, "Apple")                 # 复用的 Apple 目录

# 显式加载本目录 .env（含 API Key / DB_URL），override=True 保证以文件为准
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

# ============ 复用 Apple 的模型与数据 ============
# 分类模型：ResNet50 -> Scab / healthy / rust
CLASSIFIER_MODEL_PATH = os.getenv(
    "APPLE_CLASSIFIER_MODEL_PATH", os.path.join(PROJECT_ROOT, "best_apple_model.pth")
)
# ⚠️ 重要修正：类别顺序必须与训练时一致。
# Apple/ 的数据集目录为 Apple/train|test/{Scab, healthy, rust}，torchvision.ImageFolder
# 按文件夹名字母序编码标签，因此真实映射为 Scab=0 / healthy=1 / rust=2。
# 原脚本 Apple/apple_pest_classify_api.py 写成 ['healthy','rust','scab']，
# 实测在 Apple/test 抽样 29 张上准确率仅 13.8%，而本顺序为 79.3%，故此处采用正确顺序。
CLASS_NAMES = ["Scab", "healthy", "rust"]
INPUT_SIZE = 224                                # ResNet50 输入尺寸
CONFIDENCE_THRESHOLD = 0.6                      # 低于该置信度输出"不确定"

# 分割模型：U-Net -> background/leaf/rust/Scab（与 Apple/seg_api.py 一致）
SEG_MODEL_PATH = os.getenv(
    "APPLE_SEG_MODEL_PATH", os.path.join(APPLE_DIR, "weights", "best_model_3.pth")
)
SEG_NUM_CLASSES = 4                             # 0=背景 1=leaf 2=rust 3=Scab
SEG_INPUT_SHAPE = [512, 512]
SEG_CLASS_NAMES = ["background", "leaf", "rust", "Scab"]

# RAG 知识库：复用 Apple/apple.txt
KNOWLEDGE_FILE = os.getenv("APPLE_KNOWLEDGE_FILE", os.path.join(APPLE_DIR, "apple.txt"))

# ============ Milvus（复用 apple_agent.ipynb 的配置） ============
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_DB_NAME = "rag_tutorial"
COLLECTION_NAME = "Appledocs"
EMBED_DIM = 1024
# ⚠️ 修正：Apple/apple_agent.ipynb 里的 "Pro/BAAI/bge-m3" 在 DashScope 上已不存在
# （TextEmbedding.call 返回 400 InvalidParameter: Model not exist）。
# 实测可用且维度为 1024（与 EMBED_DIM 一致）的模型为 text-embedding-v3 / v4，
# 这里选用 text-embedding-v3。
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "text-embedding-v3")
RETRIEVE_LIMIT = 3
# COSINE 距离，越小越相关。注意：text-embedding-v3 的距离分布比 bge-m3 更平缓，
# 短查询对长文本的相似度普遍在 0.4~0.75（即相似度 0.25~0.6），
# 阈值设 0.4 会导致全部被过滤，实测 0.70 时召回正常且仍能区分相关度。
RETRIEVE_SCORE_THRESHOLD = float(os.getenv("RETRIEVE_SCORE_THRESHOLD", "0.70"))

# ============ LLM（通义千问，OpenAI 兼容接口） ============
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen-plus")
LLM_API_KEY = os.getenv("QIANWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
LLM_BASE_URL = os.getenv("QIANWEN_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# ============ PostgreSQL 持久化（LangGraph Checkpointer） ============
# LangGraph 会把每一轮的对话状态（消息、图片路径、分类结果等）写入该库
DB_URL = os.getenv(
    "DB_URL",
    "postgresql://langgraph_user:123456@localhost:5432/langgraph_db?sslmode=disable",
)

# ============ 服务 ============
AGENT_SERVICE_HOST = "0.0.0.0"
AGENT_SERVICE_PORT = int(os.getenv("AGENT_SERVICE_PORT", "8002"))
