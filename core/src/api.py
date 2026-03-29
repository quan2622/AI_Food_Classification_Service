from contextlib import asynccontextmanager
from threading import Lock
import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    CLASS_NAMES,
    DATABASE_URL,
    DEFAULT_MODEL_NAME,
    MODELS_CONFIG,
    MODEL_SAVE_DIR,
    REVIEWED_DIR,
    UPLOADS_DIR,
)
from src.db import (  # noqa: E402
    check_db_connection,
    create_prediction_log,
    get_feedback_stats,
    get_prediction_log,
    init_db,
    update_prediction_feedback,
)
from src.predict import (  # noqa: E402
    load_available_models,
    load_model_for_inference,
    predict_with_model,
    warmup_model,
)
from src.storage import (  # noqa: E402
    ensure_runtime_dirs,
    save_reviewed_copy,
    save_uploaded_file,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_runtime_dirs()
    app.state.db_connected = False
    print("FastAPI startup: dang khoi tao ket noi PostgreSQL...")
    print(f"FastAPI startup: DATABASE_URL = {DATABASE_URL}")
    init_db()
    check_db_connection()
    app.state.db_connected = True
    print("FastAPI startup: PostgreSQL connected successfully.")
    app.state.model_cache = load_available_models()
    app.state.model_locks = {model_name: Lock() for model_name in app.state.model_cache.keys()}
    for model_name, model in app.state.model_cache.items():
        warmup_model(model, model_name=model_name)
    print(
        f"FastAPI startup: loaded models -> "
        f"{list(app.state.model_cache.keys()) or 'khong co model nao trong cache'}"
    )
    yield


app = FastAPI(
    title='Vietnamese Food Classification API',
    version='1.0.0',
    description='Upload anh mon an de test model phan loai.',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class FeedbackPayload(BaseModel):
    image_id: str
    confirmed_label: str
    is_correct: bool = True
    notes: str | None = None


@app.get('/')
def root():
    return {
        'message': 'Food classification API is running.',
        'default_model': DEFAULT_MODEL_NAME,
        'available_models': list(MODELS_CONFIG.keys()),
        'num_classes': len(CLASS_NAMES),
        'runtime_storage': {
            'uploads_dir': str(UPLOADS_DIR),
            'reviewed_dir': str(REVIEWED_DIR),
            'metadata_backend': 'postgresql',
        },
    }


@app.get('/health')
def health():
    default_weight = MODEL_SAVE_DIR / f'best_{DEFAULT_MODEL_NAME}.pth'
    ensure_runtime_dirs()
    return {
        'status': 'ok',
        'default_model': DEFAULT_MODEL_NAME,
        'model_ready': default_weight.exists(),
        'num_classes': len(CLASS_NAMES),
        'uploads_ready': UPLOADS_DIR.exists(),
        'reviewed_ready': REVIEWED_DIR.exists(),
        'metadata_backend': 'postgresql',
        'db_connected': getattr(app.state, 'db_connected', False),
        'loaded_models': list(app.state.model_cache.keys()),
    }


@app.get('/classes')
def classes():
    return {'classes': CLASS_NAMES}


@app.get('/feedback/stats')
def feedback_stats():
    ensure_runtime_dirs()
    return get_feedback_stats(CLASS_NAMES)


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

    try:
        model_cache = app.state.model_cache
        model_locks = app.state.model_locks
        model = model_cache.get(model_name)
        if model is None:
            model = load_model_for_inference(model_name)
            model_cache[model_name] = model
            model_locks[model_name] = Lock()
            warmup_model(model, model_name=model_name)

        image_id, saved_path = save_uploaded_file(file.file, file.filename)
        result = predict_with_model(
            str(saved_path),
            model,
            model_name=model_name,
            model_lock=model_locks.get(model_name),
        )
        metadata = {
            'image_id': image_id,
            'original_filename': file.filename,
            'stored_path': str(saved_path),
            'uploaded_at': datetime.now(),
            'model_name': model_name,
            'top1_class_name': result['top1']['class_name'],
            'top1_confidence': result['top1']['confidence'],
            'predictions': result['predictions'],
            'is_feedback_received': False,
            'confirmed_label': None,
            'is_correct': None,
            'notes': None,
            'reviewed_path': None,
            'reviewed_at': None,
        }
        create_prediction_log(metadata)

        result['filename'] = file.filename
        result['image_id'] = image_id
        result['stored_path'] = str(saved_path)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        file.file.close()


@app.post('/feedback')
def save_feedback(payload: FeedbackPayload):
    if payload.confirmed_label not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail='confirmed_label không hợp lệ.')

    try:
        metadata = get_prediction_log(payload.image_id)
        predicted_label = metadata['top1_prediction']['class_name']
        if payload.is_correct and payload.confirmed_label != predicted_label:
            raise HTTPException(
                status_code=400,
                detail='confirmed_label phải trùng top1_prediction khi is_correct=true.',
            )

        reviewed_path = save_reviewed_copy(
            payload.confirmed_label,
            metadata['stored_path'],
        )
        update_prediction_feedback(
            payload.image_id,
            payload.confirmed_label,
            payload.is_correct,
            payload.notes,
            str(reviewed_path),
        )

        return {
            'message': 'Đã lưu feedback và tách ảnh sang reviewed dataset.',
            'image_id': payload.image_id,
            'confirmed_label': payload.confirmed_label,
            'is_correct': payload.is_correct,
            'reviewed_path': str(reviewed_path),
            'metadata_backend': 'postgresql',
        }
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
