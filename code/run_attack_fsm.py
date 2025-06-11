# run_attack_fsm.py
# FSM 策略測試主流程-測試 rewrite 策略效果
from src.utils.config_loader import load_config
from src.utils.data_loader import load_dataset
from src.utils.logger import AttackLogger
from models.opensource.qwen.qwen_wrapper import QwenModelWrapper
from src.core.state_machine import StateMachine
#reasoning,roleplay,educational
# 讀取設定
config = load_config("configs/experiment_config.yaml")
exp = config["experiment"]
log_cfg = config["logging"]

# 建立模型與資料
model = QwenModelWrapper(exp["model_name"])
dataset = load_dataset(exp["dataset_path"])[:exp["max_samples"]]

fsm = StateMachine(model, strategy=exp["strategy"])

logger = AttackLogger(
    model_name=exp["model_tag"],
    strategy=exp["strategy"],
    dataset=exp["dataset_tag"],
    base_dir=log_cfg["base_dir"],
    config_dict=config  # ⬅️ 將整份實驗設定寫入 config.json
)

for sample in dataset[:3]:
    result = fsm.run(sample["instruction"])

    print("\n============================")
    print(f"[Prompt] {result['original_prompt']}")
    print(f"[Response] {result['original_response'][:300]}...\n")
    if result["status"] == "rewritten_attempted":
        print(f"[Rewritten Prompt] {result['rewritten_prompt']}")
        print(f"[Rewritten Response] {result['rewritten_response'][:300]}...\n")

    # ✅ 判斷成功與否
    if result["status"] == "success_on_first":
        result["jailbreak_success"] = not fsm._is_rejected(result["original_response"])
    elif result["status"] == "rewritten_attempted":
        result["jailbreak_success"] = not fsm._is_rejected(result["rewritten_response"])

    # ✅ 寫入 log
    logger.log(result)

# ✅ 儲存所有 log 與 config
if log_cfg["enable_save"]:
    logger.save()
    print(f"📁 Log saved to: {logger.get_folder_path()}")