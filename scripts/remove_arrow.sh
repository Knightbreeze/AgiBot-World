# 1. 进入缓存目录（验证路径）
cd /root/.cache/huggingface/datasets/parquet

# 2. 列出即将删除的文件（预览确认）
echo "[将删除以下文件]"
find . -name "*.arrow" -exec ls -lh {} \;

# 3. 执行删除（保留目录结构）
find . -type f -name "*.arrow" -exec rm -v {} \;

# 4. 验证删除结果
echo "[删除后剩余文件]"
find . -name "*.arrow" | wc -l  # 应该输出0