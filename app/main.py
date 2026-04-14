from __future__ import annotations

import io
import os
import time
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

app = FastAPI(
    title="PetMediScan AI Skin API",
    version="1.0.0",
    description="반려동물 피부질환 진단 AI 모델 API"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 질환 정보
DISEASES = {
    "Papule&plaque": {
        "name": "구진, 플라크",
        "description": "피부에 작은 돌기(구진) 또는 넓게 융기된 병변(플라크)이 나타나며, 염증, 알레르기, 감염 등 다양한 원인으로 발생합니다.",
        "treatment": "원인에 따라 항염증제, 항히스타민제, 항생제 등의 치료가 필요할 수 있습니다.",
    },
    "Scale&Dandruff&EpidermalCollarette": {
        "name": "비듬, 각질, 상피성 잔고리",
        "description": "각질 탈락(비듬)이나 피부 표면의 비늘 모양 변화, 또는 세균 감염 후 나타나는 원형의 각질 고리(상피성 잔고리)가 관찰됩니다.",
        "treatment": "피부 감염 여부에 따라 항생제, 항진균제 또는 약용 샴푸 사용이 필요할 수 있습니다.",
    },
    "Lichenification&Hyperpigmentation": {
        "name": "태선화, 과다색소침착",
        "description": "만성적인 자극이나 긁음으로 피부가 두꺼워지고(태선화), 색소가 짙어지는 변화가 나타납니다.",
        "treatment": "가려움 조절(항히스타민제), 스테로이드 치료 및 원인 질환 관리가 필요합니다.",
    },
    "Pustule&Acne": {
        "name": "농포, 여드름",
        "description": "고름이 찬 작은 물집(농포)이나 모낭 염증으로 인한 여드름 형태의 병변이 나타납니다. 주로 세균 감염과 관련됩니다.",
        "treatment": "항생제 치료 및 피부 청결 관리가 필요할 수 있습니다.",
    },
    "Erosion&Ulcer": {
        "name": "미란, 궤양",
        "description": "피부 표면이 손상된 상태(미란) 또는 더 깊게 조직이 파괴된 병변(궤양)이 나타납니다.",
        "treatment": "상처 관리, 감염 예방을 위한 항생제 및 원인 질환 치료가 필요합니다.",
    },
    "Nodule&mass": {
        "name": "결절, 종괴",
        "description": "피부 아래 또는 표면에 단단한 덩어리 형태의 병변이 나타나며, 염증, 낭종, 종양 등 다양한 원인이 있습니다.",
        "treatment": "정확한 진단을 위해 검사(세포검사 등)가 필요하며, 필요 시 외과적 제거가 고려됩니다.",
    },
    "normal": {
        "name": "정상",
        "description": "피부에 특별한 이상 소견이 관찰되지 않는 건강한 상태입니다.",
        "treatment": "정기적인 관찰과 예방적 관리가 권장됩니다.",
    },
}

SERVICE_TYPE = os.getenv("SERVICE_TYPE", "skin")  # skin | eye
MODEL_NAME = os.getenv("MODEL_NAME", "yolov5")
MODEL_VERSION = os.getenv("MODEL_VERSION", "skin_model_final")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", MODEL_VERSION, "best.pt"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.7"))
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "50"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))

_model: Optional[YOLO] = None
_model_load_error: Optional[str] = None


def _load_model() -> None:
    global _model, _model_load_error

    if _model is not None:
        return

    if not os.path.exists(MODEL_PATH):
        _model_load_error = f"Model weights not found at '{MODEL_PATH}'"
        return

    try:
        # YOLOv5 weights may load via YOLOv5 codepath (`models.yolo`).
        # Some combos call `BaseModel.fuse(verbose=...)` while YOLOv5's implementation
        # does not accept the `verbose` kwarg. Patch it to be compatible.
        try:
            from models.yolo import BaseModel as _YoloV5BaseModel  # type: ignore

            if hasattr(_YoloV5BaseModel, "fuse"):
                _orig_fuse = _YoloV5BaseModel.fuse

                def _fuse_compat(self, *args, **kwargs):  # type: ignore
                    return _orig_fuse(self)

                _YoloV5BaseModel.fuse = _fuse_compat  # type: ignore
        except Exception:
            pass

        _model = YOLO(MODEL_PATH)
        _model_load_error = None
    except Exception as e:
        _model_load_error = str(e)


@app.on_event("startup")
def _on_startup() -> None:
    _load_model()


def _get_model_names() -> dict[int, str]:
    if _model is None:
        return {}
    names: Any = getattr(_model, "names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def _normalize_bbox_xyxy(xyxy: np.ndarray, w: int, h: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in xyxy.tolist()]
    if w <= 0 or h <= 0:
        return [x1, y1, x2, y2]
    return [x1 / w, y1 / h, x2 / w, y2 / h]


@app.get("/")
def root():
    """API 루트"""
    return {
        "service": "PetMediScan AI Skin",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
def health_check():
    """헬스체크"""
    model_loaded = _model is not None
    return {
        "status": "ok" if model_loaded else "degraded",
        "service": f"{SERVICE_TYPE}-disease-detection",
        "model_loaded": model_loaded,
        "model": {
            "type": SERVICE_TYPE,
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "path": MODEL_PATH,
        },
        "error": _model_load_error,
    }

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    피부질환 진단 API
    
    - **image**: 반려동물 피부 이미지 파일
    
    Returns:
        - disease: 진단된 질환명
        - confidence: 신뢰도 (0~1)
        - description: 질환 설명
        - treatment: 치료 방법
    """
    try:
        _load_model()
        if _model is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Model not loaded",
                    "detail": _model_load_error,
                },
            )

        t0 = time.perf_counter()
        contents = await image.read()
        t1 = time.perf_counter()

        try:
            pil = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})

        w, h = pil.size
        arr = np.array(pil)
        t2 = time.perf_counter()

        results = _model.predict(
            source=arr,
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=MAX_DETECTIONS,
            verbose=False,
        )
        t3 = time.perf_counter()

        names = _get_model_names()
        detections: list[dict[str, Any]] = []

        r0 = results[0]
        if getattr(r0, "boxes", None) is not None and len(r0.boxes) > 0:
            boxes = r0.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(conf)):
                label = names.get(int(cls[i]), str(int(cls[i])))
                detections.append(
                    {
                        "label": label,
                        "score": float(conf[i]),
                        "bbox": _normalize_bbox_xyxy(xyxy[i], w=w, h=h),
                    }
                )

        detections.sort(key=lambda d: d["score"], reverse=True)

        predicted_label = detections[0]["label"] if detections else "normal"
        predicted_score = float(detections[0]["score"]) if detections else 0.0

        disease_info = DISEASES.get(str(predicted_label), DISEASES["normal"])
        t4 = time.perf_counter()

        return {
            "model": {"type": SERVICE_TYPE, "name": MODEL_NAME, "version": MODEL_VERSION},
            "predicted": {"label": predicted_label, "score": round(predicted_score, 4)},
            "detections": detections,
            "timing_ms": {
                "read": int((t1 - t0) * 1000),
                "decode": int((t2 - t1) * 1000),
                "inference": int((t3 - t2) * 1000),
                "postprocess": int((t4 - t3) * 1000),
            },
            "image": {"width": w, "height": h},
            "disease": {
                "display_name": disease_info["name"],
                "description": disease_info["description"],
                "treatment": disease_info["treatment"],
            },
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
