#!/bin/bash

target_dir="/fs-computility/efm/shared/datasets/agibot-world/AgiBotWorld-Alpha/main"
exclude_file="${target_dir}/sample_dataset.tar"

total=0
processed=0

# 第一阶段：统计文件总数
echo "[1/2] 正在扫描压缩文件..."
find "$target_dir" -type f \( -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' -o -name '*.tar.bz2' -o -name '*.zip' \) -print0 | while IFS= read -r -d $'\0' _; do
    ((total++))
done

# 第二阶段：处理文件
echo "[2/2] 开始解压 (共发现 ${total} 个压缩文件)"
find "$target_dir" -type f \( -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' -o -name '*.tar.bz2' -o -name '*.zip' \) -print0 | while IFS= read -r -d $'\0' filepath; do
    if [[ "$filepath" == "$exclude_file" ]]; then
        continue
    fi

    ((processed++))
    dir=$(dirname "$filepath")
    filename=$(basename "$filepath")
    
    # 显示进度信息
    echo "[$(date +'%T')] 正在处理 (${processed}/${total}): ${filename}"
    
    (
        cd "$dir" || exit 1
        case "$filename" in
            *.tar)       tar xf  "$filename"  ;;
            *.tar.gz)    tar xzf "$filename"  ;;
            *.tgz)       tar xzf "$filename"  ;;
            *.tar.bz2)   tar xjf "$filename"  ;;
            *.zip)       unzip -q "$filename" ;;
        esac
    ) && echo "[$(date +'%T')] 已完成 (${processed}/${total}): ${filename}"
done

echo "所有压缩文件处理完成！"


# 赋予执行权限：chmod +x unzip_script.sh
# 执行脚本：./unzip_script.sh