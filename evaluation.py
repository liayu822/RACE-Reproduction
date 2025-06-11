# evaluation.py
import os
import json
from glob import glob
import numpy as np
from rich import print
from rich.table import Table

# 修改這裡為你的結果儲存目錄
RESULTS_DIR = "/home/server/LiangYu/RACE/results/qwen_selfplay_advbench_20250611_023703"  # 根據實際路徑調整

def load_results(result_dir: str):
    result_dir = os.path.abspath(result_dir)
    files = sorted(glob(os.path.join(result_dir, "sample_*.json")))  # ✅ 限定格式
    all_data = []
    for f in files:
        with open(f, "r") as j:
            try:
                all_data.append(json.load(j))
            except json.JSONDecodeError:
                print(f"[red]⚠️ JSON decode error:[/red] {f}")
    return all_data



def evaluate(results):
    num_total = len(results)
    num_success = sum(1 for r in results if r["jailbreak_success"])
    success_rate = num_success / num_total if num_total > 0 else 0

    # 平均資訊增益
    info_gains = []
    round_counts = []
    for r in results:
        rounds = r.get("rounds", [])
        round_counts.append(len(rounds))
        for step in rounds:
            if step["information_gain"] is not None:
                info_gains.append(step["information_gain"])

    avg_info_gain = np.mean(info_gains) if info_gains else 0
    avg_rounds = np.mean(round_counts) if round_counts else 0

    return {
        "total": num_total,
        "success": num_success,
        "success_rate": round(success_rate * 100, 2),
        "avg_info_gain": round(avg_info_gain, 4),
        "avg_rounds": round(avg_rounds, 2)
    }

def show_summary(stats):
    table = Table(title="📊 Self-Play Evaluation Summary", show_lines=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right", style="bold yellow")

    table.add_row("Total Samples", str(stats["total"]))
    table.add_row("Successful Jailbreaks", str(stats["success"]))
    table.add_row("Success Rate (%)", f"{stats['success_rate']:.2f}")
    table.add_row("Avg. Info Gain", f"{stats['avg_info_gain']:.4f}")
    table.add_row("Avg. Rounds Used", f"{stats['avg_rounds']:.2f}")

    print(table)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("[red]❌ Usage: python evaluation.py <results_dir>[/red]")
        exit(1)

    RESULTS_DIR = sys.argv[1]

    print(f"[bold green]🔍 Loading results from:[/bold green] {RESULTS_DIR}")
    results = load_results(RESULTS_DIR)
    if not results:
        print(f"[red]❌ No result files found in:[/red] {RESULTS_DIR}")
    else:
        stats = evaluate(results)
        show_summary(stats)

