import os
import glob
import pyarrow.parquet as pq
from tqdm import tqdm

# 目标数据集路径
dataset_path = "/fs-computility/efm/shared/datasets/agibot-world/AgiBotWorld-Alpha/0412_agibotworld_alpha_lerobotv21/agibotworld"

# 存储问题文件
problem_files = []

# 查找所有子文件夹下的data/chunk-*/*.parquet文件（包括chunk-000, chunk-001等所有chunk）
parquet_pattern = os.path.join(dataset_path, "**", "data", "chunk-*", "*.parquet")
parquet_files = glob.glob(parquet_pattern, recursive=True)

print(f"找到{len(parquet_files)}个parquet文件")

# 检查每个文件
for parquet_file in tqdm(parquet_files):
    try:
        # 读取parquet文件的schema
        schema = pq.read_schema(parquet_file)
        
        # 检查schema中是否包含action字段
        if 'action' not in schema.names:
            problem_files.append(parquet_file + " (缺少action字段)")
        
        # 可选：如果想验证action数据不为空，可以读取实际数据
        table = pq.read_table(parquet_file)
        if 'action' in table.column_names and len(table['action']) == 0:
            problem_files.append(parquet_file + " (action字段为空)")
            
    except Exception as e:
        problem_files.append(parquet_file + f" (错误: {str(e)})")

# 输出结果
if problem_files:
    print(f"发现 {len(problem_files)} 个问题文件:")
    for f in problem_files[:10]: 
        print(f"  - {f}")
    if len(problem_files) > 10: 
        print(f"  ...等 {len(problem_files)-10} 个文件")
else:
    print("所有parquet文件中都包含action字段并且非空!")

# 打印问题文件所在目录的汇总统计
if problem_files:
    problematic_dirs = {}
    for file in problem_files:
        parent_dir = os.path.dirname(os.path.dirname(file))  # 获取chunk-*的父目录
        if parent_dir in problematic_dirs:
            problematic_dirs[parent_dir] += 1
        else:
            problematic_dirs[parent_dir] = 1
    
    print("\n问题文件所在目录统计:")
    for dir_path, count in sorted(problematic_dirs.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {dir_path}: {count}个问题文件")