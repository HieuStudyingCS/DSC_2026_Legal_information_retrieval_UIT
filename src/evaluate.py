import os
import sys
import json
import numpy as np
import argparse

def evaluate_retrieval(y_pred: dict, y_true: dict) -> dict:
    """
    Đánh giá hiệu năng Retrieval theo đúng công thức và luật của BTC (scoring.py).
    
    :param y_pred: Dict chứa kết quả dự đoán dạng:
                   { "question_id": {"question": "...", "answer": ["doc_id1", "doc_id2"]} }
                   hoặc { "question_id": ["doc_id1", "doc_id2"] }
    :param y_true: Dict chứa đáp án gốc dạng:
                   { "question_id": {"question": "...", "answer": ["doc_id1"]} }
                   hoặc { "question_id": ["doc_id1"] }
    :return: Dict chứa {'precision': float, 'recall': float, 'total_questions': int, 'rule_violations': int}
    """
    # Chuẩn hóa định dạng y_pred về Dict[str, List[str]]
    pred_map = {}
    for k, v in y_pred.items():
        if isinstance(v, dict) and 'answer' in v:
            pred_map[str(k)] = [str(x) for x in (v['answer'] or [])]
        elif isinstance(v, list):
            pred_map[str(k)] = [str(x) for x in v]
        else:
            pred_map[str(k)] = []
            
    # Chuẩn hóa định dạng y_true về Dict[str, List[str]]
    truth_map = {}
    for k, v in y_true.items():
        if isinstance(v, dict) and 'answer' in v:
            truth_map[str(k)] = [str(x) for x in (v['answer'] or [])]
        elif isinstance(v, list):
            truth_map[str(k)] = [str(x) for x in v]
        else:
            truth_map[str(k)] = []

    recalls = []
    precisions = []
    rule_violations = 0
    empty_preds = 0

    for q_id, target_docs in truth_map.items():
        predicted_docs = pred_map.get(q_id, [])
        num_preds = len(predicted_docs)
        num_targets = len(target_docs)
        
        # LUẬT BTC: Nếu danh sách dự đoán rỗng hoặc có > 5 doc_ids -> Điểm = 0
        if num_preds == 0 or num_preds > 5 or num_targets == 0:
            recalls.append(0.0)
            precisions.append(0.0)
            if num_preds > 5:
                rule_violations += 1
            if num_preds == 0:
                empty_preds += 1
        else:
            intersection = set(target_docs) & set(predicted_docs)
            r = len(intersection) / num_targets
            p = len(intersection) / num_preds
            recalls.append(r)
            precisions.append(p)

    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    mean_precision = float(np.mean(precisions)) if precisions else 0.0

    return {
        "recall@5": mean_recall,
        "precision@5": mean_precision,
        "total_questions": len(truth_map),
        "rule_violations": rule_violations,
        "empty_predictions": empty_preds
    }

def main():
    parser = argparse.ArgumentParser(description="Đánh giá Cục bộ (Local Evaluation) cho LegalIR Task 1.")
    parser.add_argument("--pred", type=str, required=True, help="Đường dẫn file dự đoán submission.json")
    parser.add_argument("--truth", type=str, default="train.json", help="Đường dẫn file đáp án gốc (mặc định: train.json)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pred):
        print(f"❌ Không tìm thấy file dự đoán: {args.pred}")
        sys.exit(1)
        
    if not os.path.exists(args.truth):
        print(f"❌ Không tìm thấy file đáp án gốc: {args.truth}")
        sys.exit(1)
        
    print(f"📊 Đang tải file dự đoán: {args.pred}")
    with open(args.pred, 'r', encoding='utf-8') as f:
        y_pred = json.load(f)
        
    print(f"📑 Đang tải file đáp án gốc: {args.truth}")
    with open(args.truth, 'r', encoding='utf-8') as f:
        y_true = json.load(f)
        
    results = evaluate_retrieval(y_pred, y_true)
    
    print("\n" + "="*50)
    print("🏆 KẾT QUẢ ĐÁNH GIÁ CỤC BỘ (LOCAL EVALUATION METRICS)")
    print("="*50)
    print(f"🔹 Mean Recall@5:      {results['recall@5']:.4f} ({results['recall@5']*100:.2f}%)")
    print(f"🔹 Mean Precision@5:   {results['precision@5']:.4f} ({results['precision@5']*100:.2f}%)")
    print(f"📋 Tổng số câu hỏi:    {results['total_questions']}")
    print(f"⚠️ Vi phạm luật (>5):  {results['rule_violations']}")
    print(f"⚠️ Dự đoán rỗng (0):   {results['empty_predictions']}")
    print("="*50)

if __name__ == "__main__":
    main()
