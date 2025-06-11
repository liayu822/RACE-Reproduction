#攻擊語句改寫器（論文有提到可能使用 rule-based 改寫）
# src/selfplay/prompt_mutator.py

# src/selfplay/prompt_mutator.py

def logic_reasoning_mutator(prompt: str) -> str:
    return (
        "As a logic reasoning exercise, consider the following scenario:\n"
        f"{prompt}"
    )
