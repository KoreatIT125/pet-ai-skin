from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image

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
    'dermatitis': {
        'name': '피부염',
        'description': '피부 염증으로 발적, 부종, 가려움증 등이 나타납니다.',
        'treatment': '항염증제 및 항생제 치료가 필요할 수 있습니다.'
    },
    'alopecia': {
        'name': '탈모',
        'description': '국소적 또는 전신적 털 손실이 발생합니다.',
        'treatment': '원인 파악 후 적절한 치료가 필요합니다. 수의사 진료 권장.'
    },
    'eczema': {
        'name': '습진',
        'description': '가려움과 발진이 나타나는 피부 질환입니다.',
        'treatment': '항히스타민제 및 스테로이드 치료가 필요할 수 있습니다.'
    },
    'fungal': {
        'name': '곰팡이 감염',
        'description': '백선 등 곰팡이에 의한 피부 감염입니다.',
        'treatment': '항진균제 치료가 필요합니다. 전염성이 있으니 주의하세요.'
    },
    'normal': {
        'name': '정상',
        'description': '특별한 이상 소견이 관찰되지 않습니다.',
        'treatment': '건강한 상태입니다. 정기적인 검진을 권장합니다.'
    }
}

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
    return {
        "status": "ok",
        "model_loaded": False,  # TODO: 실제 모델 로드 후 True로 변경
        "service": "skin-disease-detection"
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
        # 이미지 읽기
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        
        # TODO: 실제 YOLOv8 모델 추론
        # 현재는 더미 데이터 반환
        predicted_class = 'dermatitis'  # 임시
        confidence = 0.82  # 임시
        
        disease_info = DISEASES.get(predicted_class, DISEASES['normal'])
        
        return {
            "disease": disease_info['name'],
            "confidence": round(confidence, 2),
            "description": disease_info['description'],
            "treatment": disease_info['treatment']
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
