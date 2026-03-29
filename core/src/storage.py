import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from config import REVIEWED_DIR, UPLOADS_DIR


def ensure_runtime_dirs():
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    REVIEWED_DIR.mkdir(parents=True, exist_ok=True)


def create_image_id():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{timestamp}_{uuid4().hex[:8]}'


def save_uploaded_file(source_file, original_filename):
    ensure_runtime_dirs()
    image_id = create_image_id()
    suffix = Path(original_filename or 'upload.jpg').suffix or '.jpg'
    saved_path = UPLOADS_DIR / f'{image_id}{suffix}'

    with saved_path.open('wb') as output_file:
        shutil.copyfileobj(source_file, output_file)

    return image_id, saved_path


def save_reviewed_copy(confirmed_label, upload_path):
    ensure_runtime_dirs()
    source_path = Path(upload_path)
    reviewed_class_dir = REVIEWED_DIR / confirmed_label
    reviewed_class_dir.mkdir(parents=True, exist_ok=True)
    reviewed_path = reviewed_class_dir / source_path.name
    shutil.copy2(source_path, reviewed_path)
    return reviewed_path
