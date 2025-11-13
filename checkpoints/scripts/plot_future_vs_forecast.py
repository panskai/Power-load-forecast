from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

predicted_path = SCRIPT_DIR / "predicted_load_future.csv"
forecast_path = PROJECT_ROOT / "data/processed/Area1_forecast copy.csv"

pred_df = pd.read_csv(predicted_path, parse_dates=["datetime"])
forecast_df = pd.read_csv(forecast_path, parse_dates=["datetime"])

merged = pd.merge(pred_df, forecast_df[["datetime", "load"]], on="datetime", how="inner")
merged.rename(columns={"load": "load_forecast"}, inplace=True)

plt.figure(figsize=(12, 4))
plt.plot(merged["datetime"].values, merged["load_forecast"].to_numpy(), label="外部负荷计划", color="tab:blue")
plt.plot(merged["datetime"].values, merged["load_predicted"].to_numpy(), label="模型预测负荷", color="tab:orange", alpha=0.75)
plt.xlabel("时间")
plt.ylabel("负荷 (MW)")
plt.title("预测负荷 vs. 外部计划负荷")
plt.legend()
plt.tight_layout()

output_path = PROJECT_ROOT / "future_vs_forecast.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"对比图已保存到: {output_path}")

