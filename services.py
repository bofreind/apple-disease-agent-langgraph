# -*- coding: utf-8 -*-
"""模型推理服务层：**直接复用 Apple/ 目录下已有的模型与预处理代码**。

复用方式（不复制粘贴推理逻辑，而是 import 原代码）：
- 分类：沿用 Apple/apple_pest_classify_api.py 的模型结构（ResNet50 改 3 分类）
        与预处理（Resize256 -> CenterCrop224 -> Normalize）以及三个类别
        ['healthy', 'rust', 'scab']。
- 分割：`sys.path` 注入 Apple/ 目录后，直接 import
        `Apple/model/unet_resnet.py` 的 `Unet` 网络，
        以及 `Apple/utils/utils.py` 的 `cvtColor / preprocess_input / resize_image`
        预处理函数，推理流程与 Apple/seg_api.py 完全一致。

设计要点：模型懒加载 + 单例缓存（进程内只加载一次），
        任何一个模型缺失都不会让整个服务起不来，只会让对应工具返回错误提示。
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Any, Optional

import config

# ------------------------------------------------------------------
# 把 Apple/ 目录加入 import 路径，从而复用 Apple/model、Apple/utils 包
# （Apple/model/unet_resnet.py 内部使用 `from model.resnet_backbone import ...`，
#   所以必须让 `model` 作为顶层包可被找到）
# ------------------------------------------------------------------
if config.APPLE_DIR not in sys.path:
    sys.path.insert(0, config.APPLE_DIR)


# ==================================================================
# 分类模型（ResNet50 / healthy-rust-scab）
# ==================================================================
_cls_lock = threading.Lock()
_cls_model = None
_cls_transform = None
_cls_device = None


def _load_classifier():
    """加载 ResNet50 分类模型（逻辑与 Apple/apple_pest_classify_api.py 一致）"""
    global _cls_model, _cls_transform, _cls_device
    if _cls_model is not None:
        return

    with _cls_lock:
        if _cls_model is not None:
            return

        import torch
        import torch.nn as nn
        import torchvision.models as models
        import torchvision.transforms as transforms

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = models.resnet50(weights=None)          # 不加载预训练权重
        model.fc = nn.Linear(model.fc.in_features, len(config.CLASS_NAMES))

        if not os.path.exists(config.CLASSIFIER_MODEL_PATH):
            raise FileNotFoundError(f"分类模型不存在: {config.CLASSIFIER_MODEL_PATH}")

        model.load_state_dict(
            torch.load(config.CLASSIFIER_MODEL_PATH, map_location=device)
        )
        model.to(device)
        model.eval()

        # 预处理与训练时保持一致
        _cls_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        _cls_device = device
        _cls_model = model


def classify_image(image_path: str) -> dict[str, Any]:
    """苹果叶片病害分类：返回 {success, class, class_index, confidence, all_probs, message}

    复用 Apple/apple_pest_classify_api.py 的推理与置信度阈值判断。
    """
    try:
        if not image_path or not os.path.exists(image_path):
            return {"success": False, "error": f"图片不存在: {image_path}"}

        import io

        import torch
        from PIL import Image

        _load_classifier()

        img = Image.open(io.BytesIO(open(image_path, "rb").read())).convert("RGB")
        img_tensor = _cls_transform(img).unsqueeze(0).to(_cls_device)

        with torch.no_grad():
            output = _cls_model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

        class_name = config.CLASS_NAMES[predicted.item()]
        score = float(confidence.item())
        all_probs = {
            config.CLASS_NAMES[i]: round(float(probs[0][i].item()), 4)
            for i in range(len(config.CLASS_NAMES))
        }

        # 置信度不足 -> 不确定（与 reference 实现一致的阈值策略）
        if score < config.CONFIDENCE_THRESHOLD:
            return {
                "success": True,
                "class": "unknown",
                "class_index": -1,
                "confidence": round(score, 4),
                "all_probs": all_probs,
                "message": (f"置信度不足（{score:.2f} < "
                            f"{config.CONFIDENCE_THRESHOLD}），无法确定类别"),
            }

        return {
            "success": True,
            "class": class_name,
            "class_index": int(predicted.item()),
            "confidence": round(score, 4),
            "all_probs": all_probs,
        }
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"分类失败: {e}"}


# ==================================================================
# 分割模型（U-Net / background-leaf-rust-Scab）
# ==================================================================
_seg_lock = threading.Lock()
_seg_model = None
_seg_device = None


def _load_segmenter():
    """加载 U-Net 分割模型（复用 Apple/model/unet_resnet.py 的 Unet）"""
    global _seg_model, _seg_device
    if _seg_model is not None:
        return

    with _seg_lock:
        if _seg_model is not None:
            return

        import torch

        from model.unet_resnet import Unet          # ← 复用 Apple 的网络定义

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(config.SEG_MODEL_PATH):
            raise FileNotFoundError(f"分割模型不存在: {config.SEG_MODEL_PATH}")

        net = Unet(num_classes=config.SEG_NUM_CLASSES)
        net.load_state_dict(
            torch.load(config.SEG_MODEL_PATH, map_location=device)
        )
        net.eval()
        net.to(device)

        _seg_device = device
        _seg_model = net


def segment_image(image_path: str) -> dict[str, Any]:
    """苹果叶片语义分割：返回各类别占比 + 病变率

    推理与统计逻辑复用 Apple/seg_api.py 的 predict_mask + 占比计算。
    类别约定：0=background 1=leaf 2=rust 3=Scab
    """
    try:
        if not image_path or not os.path.exists(image_path):
            return {"success": False, "error": f"图片不存在: {image_path}"}

        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image

        from utils.utils import cvtColor, preprocess_input, resize_image  # ← 复用 Apple 预处理

        _load_segmenter()

        image = Image.open(image_path).convert("RGB")
        image = cvtColor(image)
        input_shape = config.SEG_INPUT_SHAPE

        image_data, nw, nh = resize_image(image, (input_shape[1], input_shape[0]))
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data).to(_seg_device)
            pr = _seg_model(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            # 裁掉 resize 时补的灰边，只保留有效内容区（nh x nw）
            top = int((input_shape[0] - nh) // 2)
            left = int((input_shape[1] - nw) // 2)
            pr = pr[top: top + nh, left: left + nw]
            # 直接在缩放后的尺寸上取类别，不再 cv2.resize 放大回原图：
            #   1) 缩放是等比例的，各类别占比与放大后统计完全一致；
            #   2) 内存从 O(原图像素数) 降到 O(512x512)。
            #      原写法对大图（如 2988x5312 ≈ 1590 万像素）会分配
            #      ~127MB 的 int64 argmax 结果 + ~254MB 的 float32 概率图而 OOM。
            mask = pr.argmax(axis=-1)

        total = int(mask.size)
        leaf_px = int((mask == 1).sum())
        rust_px = int((mask == 2).sum())
        scab_px = int((mask == 3).sum())
        disease_px = rust_px + scab_px

        leaf_ratio = leaf_px / total if total else 0.0
        rust_ratio = rust_px / total if total else 0.0
        scab_ratio = scab_px / total if total else 0.0
        disease_ratio = disease_px / (leaf_px + disease_px) if (leaf_px + disease_px) else 0.0

        return {
            "success": True,
            "leaf_ratio": round(leaf_ratio, 4),
            "rust_ratio": round(rust_ratio, 4),
            "scab_ratio": round(scab_ratio, 4),
            "disease_ratio": round(disease_ratio, 4),
            "has_disease": disease_px > 0,
            "has_leaf": leaf_px > 0,
            "pixels": {"leaf": leaf_px, "rust": rust_px, "Scab": scab_px, "total": total},
        }
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"分割失败: {e}"}


GEO_URL  = "https://geocoding-api.open-meteo.com/v1/search"
FCST_URL = "https://api.open-meteo.com/v1/forecast"

def get_weather(city: str, days: int = 3) -> dict:
    """查询城市天气（Open-Meteo，免密钥）。返回 {success, place, current, daily}"""
    proxies = {"http": None, "https": None}      # 硬清代理，防止环境里的死代理
    try:
        import requests
        g = requests.get(GEO_URL, timeout=8, proxies=proxies,
                         params={"name": city, "count": 1, "language": "zh", "format": "json"})
        res = g.json().get("results")
        if not res:
            return {"success": False, "error": f"没找到地名「{city}」"}
        lat, lon = res[0]["latitude"], res[0]["longitude"]

        f = requests.get(FCST_URL, timeout=8, proxies=proxies, params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
            "timezone": "Asia/Shanghai", "forecast_days": days,
        })
        return {"success": True,
                "place": f"{res[0]['name']}（{res[0].get('admin1')}）",
                "raw": f.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}
