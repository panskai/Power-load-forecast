FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 默认命令：你可以换成自己的推理脚本，比如 predict_future_load.py
CMD ["python", "scripts/predict_future_load.py"]
