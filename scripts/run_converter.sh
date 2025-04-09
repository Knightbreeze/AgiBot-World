#!/bin/bash
# run_converter.sh - 在后台运行数据集转换器

# 设置数据路径
SRC_PATH="/fs-computility/efm/shared/datasets/agibot-world/AgiBotWorld-Alpha/sample_dataset"
TGT_PATH="/fs-computility/efm/shared/datasets/agibot-world"
REPO_ID="lerobotV2_AgiBotWorld_sample"

# 创建日志目录
mkdir -p "$TGT_PATH/logs"

# 设置时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$TGT_PATH/logs/convert_$TIMESTAMP.log"

# 运行转换脚本，使用nohup保持在后台运行
nohup python -u /fs-computility/efm/kongweijie/AgiBot-World/scripts/distributed_convert_agibot2lerobot.py \
  --src_path "$SRC_PATH" \
  --tgt_path "$TGT_PATH" \
  --repo_id "$REPO_ID" \
  --chunk_size 20 \
  > "$LOG_FILE" 2>&1 &

echo "转换任务已在后台启动, 进程ID: $!"
echo "日志文件: $LOG_FILE"
echo "使用 'tail -f $LOG_FILE' 查看进度"