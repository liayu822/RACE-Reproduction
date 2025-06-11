# src/core/attack_state_machine.py
from src.modules.rejection_detector import RejectionDetector

class AttackStateMachine:
    def __init__(self, target_query: str, conversation_limit: int = 3):
        self.states = ["initial"] + [f"s{i}" for i in range(1, conversation_limit)] + ["success", "failure"]
        self.current_state = "initial"
        self.query = target_query
        self.history = []
        self.max_rounds = conversation_limit
        self.detector = RejectionDetector()

    def transition(self, prompt: str, response: str) -> str:
        self.history.append({"prompt": prompt, "response": response})
        round_index = len(self.history)

        if not self.detector.is_rejection(response):
            self.current_state = "success"
        elif round_index >= self.max_rounds:
            self.current_state = "failure"
        else:
            self.current_state = f"s{round_index}"

        return self.current_state

    def is_terminal(self) -> bool:
        return self.current_state in {"success", "failure"}

    def is_success(self) -> bool:
        return self.current_state == "success"

    def get_history(self):
        return self.history

    def get_final_response(self):
        return self.history[-1]["response"] if self.history else ""
