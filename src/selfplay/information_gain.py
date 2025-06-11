# src/selfplay/information_gain.py
#計算 query 與回應的資訊增益
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")
model.save("./sentence_model")
from sklearn.metrics.pairwise import cosine_similarity
import torch

class InfoGainScorer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = SentenceTransformer("./sentence_model", device=self.device)

    def compute_gain(self, prev_response: str, new_response: str) -> float:
        """
        計算語意資訊增益（1 - cosine similarity）
        """
        emb1 = self.encoder.encode(prev_response, convert_to_tensor=True)
        emb2 = self.encoder.encode(new_response, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb1, emb2).item()
        return round(1.0 - similarity, 4)

    def batch_compute(self, reference: str, candidates: list[str]) -> list[float]:
        """
        批次計算 reference 與多個回應的資訊增益
        """
        emb_ref = self.encoder.encode(reference, convert_to_tensor=True)
        emb_cands = self.encoder.encode(candidates, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(emb_ref, emb_cands)
        return [round(1.0 - s.item(), 4) for s in scores[0]]


# 範例用法
if __name__ == "__main__":
    scorer = InfoGainScorer()
    r1 = "I'm sorry, but I can't help with that."
    r2 = "Here are 5 ways attackers usually try to breach a government database..."
    score = scorer.compute_gain(r1, r2)
    print(f"Info Gain: {score:.4f}")
