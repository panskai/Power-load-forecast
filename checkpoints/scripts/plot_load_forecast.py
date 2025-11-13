from pathlib import Path
import sys
import json
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.lstm_transformer import LSTMTransformerModel
from utils.dataset import LoadDataset

area = "Area1"
checkpoints_root = Path("checkpoints/lstm_transformer")
checkpoint_dir = sorted(checkpoints_root.glob(f"lstm_transformer_{area}_*"))[-1]
model_path = checkpoint_dir / "best_model.pth"
config_path = checkpoint_dir / "config.json"

with config_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

dataset = LoadDataset(
    data_path=f"data/processed/{area}_test.csv",
    sequence_length=cfg.get("sequence_length", 96),
    use_time_features=cfg.get("use_time_features", True)
)

model = LSTMTransformerModel(
    input_dim=cfg.get("input_dim", len(dataset.feature_columns)),
    lstm_hidden_dim=cfg.get("lstm_hidden_dim", 128),
    lstm_layers=cfg.get("lstm_layers", 2),
    transformer_dim=cfg.get("transformer_dim", 32),
    transformer_layers=cfg.get("transformer_layers", 4),
    num_heads=cfg.get("num_heads", 8),
    sequence_length=cfg.get("sequence_length", dataset.sequence_length),
    output_dim=1,
    dropout=cfg.get("dropout", 0.1)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

state = torch.load(model_path, map_location="cpu")
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]
model.load_state_dict(state)
model.eval()

batch_size = 256
predictions = []
with torch.no_grad():
    for start in range(0, len(dataset), batch_size):
        end = start + batch_size
        batch = torch.from_numpy(dataset.sequences[start:end]).float().to(device)
        outputs, _ = model(batch)
        predictions.append(outputs.cpu().numpy().flatten())

pred_scaled = np.concatenate(predictions)
pred_load = dataset.inverse_transform_target(pred_scaled)

df = pd.DataFrame({
    "datetime": dataset.data["datetime"][96:].reset_index(drop=True),
    "load_actual": dataset.data["load"][96:].reset_index(drop=True),
    "load_predicted": pred_load
})

output_dir = Path(__file__).resolve().parent
output_dir.mkdir(parents=True, exist_ok=True)

fig_actual, ax_actual = plt.subplots(figsize=(12, 4))
ax_actual.plot(df["datetime"], df["load_actual"], color="tab:blue", label="实际负荷")
ax_actual.set_xlabel("时间")
ax_actual.set_ylabel("负荷 (MW)")
ax_actual.set_title(f"{area} 测试集原始负荷曲线")
ax_actual.legend()
fig_actual.tight_layout()
actual_path = output_dir / f"forecast_actual_{area}.png"
fig_actual.savefig(actual_path, dpi=300)
plt.close(fig_actual)

fig_pred, ax_pred = plt.subplots(figsize=(12, 4))
ax_pred.plot(df["datetime"], df["load_predicted"], color="tab:orange", label="预测负荷")
ax_pred.set_xlabel("时间")
ax_pred.set_ylabel("负荷 (MW)")
ax_pred.set_title(f"{area} 测试集预测负荷曲线")
ax_pred.legend()
fig_pred.tight_layout()
pred_path = output_dir / f"forecast_predicted_{area}.png"
fig_pred.savefig(pred_path, dpi=300)
plt.close(fig_pred)

fig_combined, ax_combined = plt.subplots(figsize=(12, 4))
ax_combined.plot(df["datetime"], df["load_actual"], label="实际负荷", color="tab:blue")
ax_combined.plot(df["datetime"], df["load_predicted"], label="预测负荷", color="tab:orange", alpha=0.75)
ax_combined.set_xlabel("时间")
ax_combined.set_ylabel("负荷 (MW)")
ax_combined.set_title(f"{area} 测试集负荷预测对比")
ax_combined.legend()
fig_combined.tight_layout()
comparison_path = output_dir / f"forecast_comparison_{area}.png"
fig_combined.savefig(comparison_path, dpi=300)

print(f"实际负荷曲线已保存至: {actual_path}")
print(f"预测负荷曲线已保存至: {pred_path}")
print(f"对比曲线已保存至: {comparison_path}")

plt.show()