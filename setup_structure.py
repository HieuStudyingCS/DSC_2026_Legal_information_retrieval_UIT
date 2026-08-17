import os
import sys
import shutil
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def reorganize_project():
    base_dir = Path(__file__).resolve().parent

    # 1. Danh sách các thư mục cần tạo mới
    directories_to_create = [
        base_dir / "data" / "raw",
        base_dir / "data" / "processed",
        base_dir / "indices" / "faiss",
        base_dir / "indices" / "bm25",
        base_dir / "models" / "embedding",
        base_dir / "models" / "reranker",
        base_dir / "src" / "retrieval",
        base_dir / "docs",
    ]

    print("Đang tạo cấu trúc thư mục mới...")
    for folder in directories_to_create:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"  └─ Created: {folder.relative_to(base_dir)}")

    # 2. Di chuyển các tệp dữ liệu thô vào data/raw/
    raw_files = ["train.json", "warmup.json", "public-official.json"]
    for file_name in raw_files:
        src_file = base_dir / file_name
        dest_file = base_dir / "data" / "raw" / file_name
        if src_file.exists():
            shutil.move(str(src_file), str(dest_file))
            print(f"Moved: {file_name} -> data/raw/")

    # 3. Di chuyển thư mục selected-contexts vào data/raw/
    src_contexts = base_dir / "selected-contexts"
    dest_contexts = base_dir / "data" / "raw" / "selected-contexts"
    if src_contexts.exists() and not dest_contexts.exists():
        shutil.move(str(src_contexts), str(dest_contexts))
        print(f"Moved: selected-contexts/ -> data/raw/selected-contexts/")

    # 4. Di chuyển dữ liệu đã xử lý vào data/processed/
    src_processed = base_dir / "processed_data"
    if src_processed.exists():
        for jsonl_file in src_processed.glob("*.jsonl"):
            shutil.move(str(jsonl_file), str(base_dir / "data" / "processed" / jsonl_file.name))
            print(f"Moved: {jsonl_file.name} -> data/processed/")
        shutil.rmtree(str(src_processed), ignore_errors=True)

    # 5. Di chuyển tài liệu vào docs/
    doc_files = ["allowed_models.md", "read_docx.ps1", "DSC2026_Task1_LegalIR_Data_Overview.docx"]
    for doc_name in doc_files:
        src_doc = base_dir / doc_name
        dest_doc = base_dir / "docs" / doc_name
        if src_doc.exists():
            shutil.move(str(src_doc), str(dest_doc))
            print(f"Moved: {doc_name} -> docs/")


if __name__ == "__main__":
    reorganize_project()
