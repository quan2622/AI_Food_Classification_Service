import os
import shutil
import argparse

# Danh sách giữ lại
KEEP_LIST = {
    "banh mi",
    "goi cuon",
    "com tam",
    "banh xeo",
    "pho",
    "banh trang nuong",
    "banh cuon",
    "ca kho to",
    "xoi xeo",
    "banh khot"
}

def clean_directories(target_path, dry_run=False):
    if not os.path.exists(target_path):
        print(f"Path không tồn tại: {target_path}")
        return

    for name in os.listdir(target_path):
        full_path = os.path.join(target_path, name)

        if os.path.isdir(full_path) and name not in KEEP_LIST:
            if dry_run:
                print(f"[DRY RUN] Sẽ xóa: {full_path}")
            else:
                print(f"Đang xóa: {full_path}")
                shutil.rmtree(full_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xóa thư mục không nằm trong whitelist")
    parser.add_argument("path", help="Đường dẫn tới folder cần lọc")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in ra, không xóa")

    args = parser.parse_args()

    clean_directories(args.path, args.dry_run)