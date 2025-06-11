# models/open_source/qwen/qwen_wrapper.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class QwenModelWrapper:
    def __init__(self, model_name="Qwen/Qwen2-7B-Instruct", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.device = device

    def chat(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=512,
                    do_sample=True,          # 預設使用 sampling，否則 greedy
                    temperature=0.7,         # 控制隨機程度（越高越亂）
                    top_p=0.9,               # nucleus sampling
                    eos_token_id=self.tokenizer.eos_token_id  # 強制遇到 <|endoftext|> 結束
                )


        # 擷取新產生部分
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
