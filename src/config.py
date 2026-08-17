import os
import sys
from pathlib import Path

# Đảm bảo in UTF-8 trên Windows PowerShell
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# =========================================================================
# 1. TỰ ĐỘNG PHÁT HIỆN MÔI TRƯỜNG CHẠY (LOCAL VS DOCKER BTC)
# =========================================================================
# Thư mục gốc dự án ở môi trường Local (DSC_2026_Legal_information_retrieval_UIT/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Đường dẫn kiểm tra môi trường Docker của BTC
BTC_DOCKER_INPUT = Path("/app/input")

# Biến cờ xác định môi trường
IS_BTC_DOCKER = BTC_DOCKER_INPUT.exists()

if IS_BTC_DOCKER:
    # ---------------------------------------------------------------------
    # CẤU HÌNH ĐƯỜNG DẪN MÔI TRƯỜNG DOCKER BTC (CODALAB)
    # ---------------------------------------------------------------------
    REF_DIR = Path("/app/input/ref")
    RES_DIR = Path("/app/input/res")
    OUTPUT_DIR = Path("/app/output")
    
    TRAIN_PATH = REF_DIR / "train.json"
    PUBLIC_TEST_PATH = RES_DIR / "public-official.json"
    SUBMISSION_OUTPUT_PATH = OUTPUT_DIR / "submission.json"
    
    DATA_RAW_DIR = REF_DIR
    DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    INDICES_DIR = PROJECT_ROOT / "indices"
    MODELS_DIR = PROJECT_ROOT / "models"
else:
    # ---------------------------------------------------------------------
    # CẤU HÌNH ĐƯỜNG DẪN MÔI TRƯỜNG LOCAL / COLAB
    # ---------------------------------------------------------------------
    DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    INDICES_DIR = PROJECT_ROOT / "indices"
    MODELS_DIR = PROJECT_ROOT / "models"
    
    TRAIN_PATH = DATA_RAW_DIR / "train.json"
    WARMUP_PATH = DATA_RAW_DIR / "warmup.json"
    PUBLIC_TEST_PATH = DATA_RAW_DIR / "public-official.json"
    SELECTED_CONTEXTS_DIR = DATA_RAW_DIR / "selected-contexts"
    SUBMISSION_OUTPUT_PATH = PROJECT_ROOT / "submission.json"

# =========================================================================
# 2. ĐƯỜNG DẪN CHỈ MỤC VÀ DỮ LIỆU ĐÃ XỬ LÝ
# =========================================================================
PROCESSED_CORPUS_JSONL = DATA_PROCESSED_DIR / "processed_corpus.jsonl"

FAISS_INDEX_PATH = INDICES_DIR / "faiss" / "dense_index.faiss"
BM25_INDEX_PATH = INDICES_DIR / "bm25" / "bm25_index.pkl"
DOC_MAPPING_PATH = INDICES_DIR / "doc_mapping.json"

# =========================================================================
# 3. THÔNG SỐ SIÊU THAM SỐ (HYPERPARAMETERS)
# =========================================================================
MAX_WORDS_PER_CHUNK = 600
OVERLAP_WORDS = 50

# Embedding Model (Local Weights Path)
EMBEDDING_MODEL_NAME_OR_PATH = os.getenv(
    "EMBEDDING_MODEL_PATH", 
    str(MODELS_DIR / "embedding" / "Vietnamese_Embedding_v2")
)

# Reranker Model (Local Weights Path)
RERANKER_MODEL_NAME_OR_PATH = os.getenv(
    "RERANKER_MODEL_PATH", 
    str(MODELS_DIR / "reranker" / "bge-reranker-large")
)

TOP_K_RETRIEVAL = 5  # Ràng buộc tối đa 5 document_id của BTC
