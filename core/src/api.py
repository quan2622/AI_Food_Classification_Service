import shutil
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import CLASS_NAMES, DEFAULT_MODEL_NAME, MODELS_CONFIG, MODEL_SAVE_DIR  # noqa: E402
from src.predict import predict  # noqa: E402

app = FastAPI(
    title='Vietnamese Food Classification API',
    version='1.0.0',
    description='Upload anh mon an de test model phan loai.',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/')
def root():
    return {
        'message': 'Food classification API is running.',
        'default_model': DEFAULT_MODEL_NAME,
        'available_models': list(MODELS_CONFIG.keys()),
        'num_classes': len(CLASS_NAMES),
    }


@app.get('/health')
def health():
    default_weight = MODEL_SAVE_DIR / f'best_{DEFAULT_MODEL_NAME}.pth'
    return {
        'status': 'ok',
        'default_model': DEFAULT_MODEL_NAME,
        'model_ready': default_weight.exists(),
        'num_classes': len(CLASS_NAMES),
    }


@app.get('/classes')
def classes():
    return {'classes': CLASS_NAMES}


@app.post('/predict')
def predict_image(
    file: UploadFile = File(...),
    model_name: str = DEFAULT_MODEL_NAME,
):
    if model_name not in MODELS_CONFIG:
        raise HTTPException(status_code=400, detail='Model name không hợp lệ.')

    model_path = MODEL_SAVE_DIR / f'best_{model_name}.pth'
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f'Chưa tìm thấy file model đã train: {model_path.name}',
        )

    suffix = Path(file.filename or 'upload.jpg').suffix or '.jpg'
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            shutil.copyfileobj(file.file, temp_file)

        result = predict(str(temp_path), model_name=model_name)
        result['filename'] = file.filename
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        file.file.close()
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
