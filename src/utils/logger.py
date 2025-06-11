# src/utils/logger.py

import os
import json
from datetime import datetime

class AttackLogger:
    def __init__(self, model_name, strategy, dataset, base_dir="results", config_dict=None):
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(base_dir, f"{model_name}_{strategy}_{dataset}_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)

        if config_dict:
            with open(os.path.join(self.save_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

        self.sample_count = 0

    def save_sample(self, result: dict):
        """
        儲存單一 self-play 的結果（包含 seed_prompt、多輪交互與最終回應）
        """
        filename = os.path.join(self.save_dir, f"sample_{self.sample_count:04d}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        self.sample_count += 1

    def log(self, result: dict):
        self.logs.append(result)

    def save(self):
        # 儲存 log 結果
        with open(self.save_path, "w", encoding="utf-8") as f:
            for item in self.logs:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

        # 儲存當次使用的設定
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved {len(self.logs)} results to: {self.save_path}")
        print(f"🛠️  Config saved to: {self.config_path}")

    def get_folder_path(self):
        return self.folder
