# test_qwen_wrapper.py

from models.opensource.qwen.qwen_wrapper import QwenModelWrapper

def main():
    model = QwenModelWrapper()
    prompt = "Give me a short introduction to large language models."
    response = model.chat(prompt)
    print(f"\n🧠 Model Response:\n{response}")

if __name__ == "__main__":
    main()
