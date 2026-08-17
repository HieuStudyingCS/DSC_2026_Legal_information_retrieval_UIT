import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# CẤU HÌNH HỆ THỐNG - DỄ THEO DÕI VÀ ĐIỀU CHỈNH

# 1. Đường dẫn Dữ liệu (Paths)
DEFAULT_INPUT_PATH = r"D:\Project Vibe Coding\DSC_2026\selected-contexts\selected-contexts\context_21.json"
DEFAULT_OUTPUT_DIR = r"processed_data"

# 2. Tham số Tiền xử lý (Preprocessing)
# Áp dụng cho Sliding Window (Lớp 2)
MAX_WORDS_PER_CHUNK = 6000  # Phù hợp với ngữ cảnh 8192 tokens
OVERLAP_WORDS = 50          # Số từ gối đầu nhau để không đứt nghĩa

# 3. Thông số Mô hình (Models)
EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding_v2"
EMBEDDING_DIM = 1024
MAX_POSITION_EMBEDDINGS = 8194


tokenizer = AutoTokenizer.from_pretrained('AITeamVN/Vietnamese_Reranker')
model = AutoModelForSequenceClassification.from_pretrained('AITeamVN/Vietnamese_Reranker')
model.eval()
MAX_LENGTH = 2304
pairs = [['Trí tuệ nhân tạo là gì?', 'Trí tuệ nhân tạo là công nghệ giúp máy móc suy nghĩ và học hỏi như con người. Nó hoạt động bằng cách thu thập dữ liệu, nhận diện mẫu và đưa ra quyết định.'],
         ['Trí tuệ nhân tạo là gì?', 'Giấc ngủ giúp cơ thể và não bộ nghỉ ngơi, hồi phục năng lượng và cải thiện trí nhớ. Ngủ đủ giấc giúp tinh thần tỉnh táo và làm việc hiệu quả hơn.']]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=MAX_LENGTH)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)

'''
# tensor([ 7.5590, -9.0743])
'''
