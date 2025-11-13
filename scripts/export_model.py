import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import boto3

REQUIRED_FILES = {
    "config.json",
    "best_model.pth",
    "training_history.json",
    "evaluation_results.json",
    "scaler.pkl",
}

OPTIONAL_FILES = {
    "final_model.pth",
    "feature_columns.json",
}


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_metadata(config: Dict[str, Any],
                   evaluation: Dict[str, Any],
                   version: str) -> Dict[str, Any]:
    utc_now = datetime.now(timezone.utc)
    metadata = {
        "model_name": "load_forecast_lstm_transformer",
        "version": version,
        "uploaded_at": utc_now.isoformat(),
        "config": config,
    }

    area = config.get("area")
    if area:
        metadata["area"] = area

    if evaluation:
        metadata["evaluation_metrics"] = evaluation.get("test_metrics") or evaluation

    return metadata


def upload_file(s3_client, local_path: Path, bucket: str, key: str, dry_run: bool):
    if dry_run:
        print(f"[dry-run] would upload {local_path} -> s3://{bucket}/{key}")
        return
    print(f"上传 {local_path} -> s3://{bucket}/{key}")
    s3_client.upload_file(str(local_path), bucket, key)


def ensure_files_exist(checkpoint_dir: Path):
    missing = [f for f in REQUIRED_FILES if not (checkpoint_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"缺少必要文件: {', '.join(missing)}，请先确认训练目录中存在这些文件。"
        )


def resolve_version(checkpoint_dir: Path, explicit_version: str = None) -> str:
    if explicit_version:
        return explicit_version
    return checkpoint_dir.name


def main():
    parser = argparse.ArgumentParser(
        description="导出负荷预测模型工件并上传至 S3"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="本地训练输出目录，需包含 config.json、best_model.pth 等文件",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        required=True,
        help="目标 S3 bucket 名称",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default="shared",
        help="S3 前缀（默认 shared）",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="模型版本标识，默认使用 checkpoint 目录名",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要执行的操作，不上传",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not checkpoint_dir.exists():
        print(f"错误: {checkpoint_dir} 不存在")
        sys.exit(1)

    ensure_files_exist(checkpoint_dir)

    config = load_json(checkpoint_dir / "config.json")
    evaluation = load_json(checkpoint_dir / "evaluation_results.json")

    area = config.get("area", "Area1")
    version = resolve_version(checkpoint_dir, args.version)

    s3_client = boto3.client("s3")
    bucket = args.s3_bucket
    prefix = args.s3_prefix.strip("/")

    pretrained_prefix = f"{prefix}/pre-trained_models/load_forecast/{area}/{version}"
    predictions_prefix = f"{prefix}/predictions/load_forecast/{area}/{version}"

    metadata = build_metadata(config, evaluation, version)
    metadata_path = checkpoint_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    upload_targets = []
    cleanup_paths = []
    for filename in REQUIRED_FILES.union(OPTIONAL_FILES).union({"metadata.json"}):
        file_path = checkpoint_dir / filename
        if file_path.exists():
            key = f"{pretrained_prefix}/{filename}"
            upload_targets.append((file_path, key))

    # 单独处理 evaluation_results.json -> predictions 路径
    evaluation_path = checkpoint_dir / "evaluation_results.json"
    if evaluation_path.exists():
        key = f"{predictions_prefix}/evaluation_results.json"
        upload_targets.append((evaluation_path, key))

    # 生成 metrics 文件
    if evaluation:
        utc_now = datetime.utcnow()
        metrics_dir = f"{prefix}/predictions/metrics/{utc_now:%Y%m%d}"
        metrics_filename = f"metrics_{int(time.time())}.json"
        metrics_data = {
            "area": area,
            "model_version": version,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "metrics": evaluation.get("test_metrics", evaluation),
        }
        metrics_path = checkpoint_dir / metrics_filename
        metrics_path.write_text(json.dumps(metrics_data, indent=2, ensure_ascii=False), encoding="utf-8")
        upload_targets.append((metrics_path, f"{metrics_dir}/{metrics_filename}"))
        cleanup_paths.append(metrics_path)

    for local_path, key in upload_targets:
        upload_file(s3_client, local_path, bucket, key, args.dry_run)

    for path in cleanup_paths:
        if path.exists():
            path.unlink()

    if not args.dry_run:
        print("上传完成！")
    else:
        print("dry-run 模式下未执行真实上传操作。")


if __name__ == "__main__":
    main()

