# src/modules/self_play.py
from src.selfplay.information_gain import InfoGainScorer
from src.selfplay.prompt_mutator import logic_reasoning_mutator
class SelfPlayManager:
    def __init__(self, model_wrapper, rejection_detector, max_rounds=3):
        self.model = model_wrapper
        self.detector = rejection_detector
        self.max_rounds = max_rounds
        self.info_scorer = InfoGainScorer()  # ⬅️ 建立一次 encoder 實例即可

    def run(self, seed_prompt: str) -> dict:
        history = []
        current_prompt = seed_prompt
        success = False

        for round_num in range(self.max_rounds):
            attacker_prompt = logic_reasoning_mutator(current_prompt)
            response = self.model.chat(attacker_prompt)

            if round_num == 0:
                info_gain = None
            else:
                info_gain = self.info_scorer.compute_gain(prev_response, response)

            history.append({
                "round": round_num + 1,
                "attacker_prompt": attacker_prompt,
                "response": response,
                "information_gain": info_gain
            })

            if not self.detector.is_rejection(response):
                success = True
                break
            else:
                current_prompt = response
                prev_response = response  # ⬅️ 儲存上一輪回應供下一輪使用


        return {
            "seed_prompt": seed_prompt,
            "rounds": history,
            "jailbreak_success": success,
            "final_response": history[-1]["response"]
        }
