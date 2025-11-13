from pathlib import Path
import sys
import json
import torch
import pandas as pd
import numpy as np
from datetime import timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.lstm_transformer import LSTMTransformerModel
from utils.dataset import LoadDataset

# 1. 选择模型与配置
area = "Area1"
checkpoint_dir = PROJECT_ROOT / "checkpoints/lstm_transformer/lstm_transformer_Area1_20251022_155417"
model_path = checkpoint_dir / "best_model.pth"
config_path = checkpoint_dir / "config.json"

cfg = json.loads(config_path.read_text(encoding="utf-8"))

# 2. 加载训练集以复用标准化器
train_dataset = LoadDataset(
    data_path=str(PROJECT_ROOT / f"data/processed/{area}_train.csv"),
    sequence_length=cfg["sequence_length"],
    use_time_features=cfg.get("use_time_features", True)
)
scaler = train_dataset.scaler
feature_cols = train_dataset.feature_columns

# 3. 读取最新历史窗口 & 未来天气预报
history = pd.read_csv(PROJECT_ROOT / "data/processed/Area1_history.csv", parse_dates=["datetime"])
forecast_weather = pd.read_csv(PROJECT_ROOT / "data/processed/Area1_forecast.csv", parse_dates=["datetime"])

for df in (history, forecast_weather):
    df["quarter_hour"] = df["datetime"].dt.hour * 4 + df["datetime"].dt.minute // 15

# 确保历史窗口长度 = sequence_length
history = history.sort_values("datetime").tail(cfg["sequence_length"]).reset_index(drop=True)

# 4. 构建模型并加载权重
model = LSTMTransformerModel(
    input_dim=cfg["input_dim"],
    lstm_hidden_dim=cfg["lstm_hidden_dim"],
    lstm_layers=cfg["lstm_layers"],
    transformer_dim=cfg["transformer_dim"],
    transformer_layers=cfg["transformer_layers"],
    num_heads=cfg["num_heads"],
    sequence_length=cfg["sequence_length"],
    output_dim=1,
    dropout=cfg["dropout"]
)

state = torch.load(model_path, map_location="cpu")
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]
model.load_state_dict(state)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 5. 迭代预测
horizon = len(forecast_weather)
results = []

current_window = history.copy()

for step in range(horizon):
    # 标准化输入
    features = current_window[feature_cols].to_numpy(dtype=np.float32)
    scaled = scaler.transform(features)
    seq = torch.from_numpy(scaled).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_scaled, _ = model(seq)
    pred_load = train_dataset.inverse_transform_target(pred_scaled.cpu().numpy())[0]

    target_time = forecast_weather.iloc[step]["datetime"]
    results.append({
        "datetime": target_time,
        "load_predicted": pred_load
    })

    # 生成下一步所需的新行：load 用预测值，其它特征取天气预报
    new_row = {
        "datetime": target_time,
        "load": pred_load,
        "temp_avg": forecast_weather.iloc[step]["temp_avg"],
        "humidity": forecast_weather.iloc[step]["humidity"],
        "rain": forecast_weather.iloc[step]["rain"],
        "quarter_hour": forecast_weather.iloc[step]["quarter_hour"]
    }
    current_window = (
        pd.concat([current_window, pd.DataFrame([new_row])], ignore_index=True)
           .iloc[-cfg["sequence_length"]:]
    )

forecast_df = pd.DataFrame(results)
output_path = PROJECT_ROOT / "checkpoints/scripts/predicted_load_future.csv"
forecast_df.to_csv(output_path, index=False)
print(forecast_df.head())
print(f"预测结果已写入 {output_path}")