# import json
# import os
# from collections import defaultdict

# # 初始化统计字典
# task_stats = defaultdict(lambda: {"episodes": 0, "skills": 0, "source_files": set()})
# task_files_count = 0
# unique_task_names = set()

# # 遍历task_info目录下的所有JSON文件
# task_info_dir = "/fs-computility/efm/shared/datasets/agibot-world/AgiBotWorld-Alpha/main/task_info"
# for filename in os.listdir(task_info_dir):
#     if filename.startswith("task_") and filename.endswith(".json"):
#         task_files_count += 1
#         filepath = os.path.join(task_info_dir, filename)
#         with open(filepath, 'r', encoding='utf-8') as f:
#             try:
#                 episodes = json.load(f)
#                 for episode in episodes:
#                     task_name = episode["task_name"]
#                     unique_task_names.add(task_name)
#                     # 记录来源文件
#                     task_stats[task_name]["source_files"].add(filename)
#                     # 统计episode数量
#                     task_stats[task_name]["episodes"] += 1
#                     # 统计skill数量
#                     skills = len(episode["label_info"]["action_config"])
#                     task_stats[task_name]["skills"] += skills
#             except json.JSONDecodeError:
#                 print(f"错误：文件 {filename} 格式无效")

# # 计算全局总和
# total_episodes = sum(stats["episodes"] for stats in task_stats.values())
# total_skills = sum(stats["skills"] for stats in task_stats.values())


# # 构建输出内容
# output_content = [
#     "===== dir =====",
#     f"1. num_task_xxx.json: {task_files_count}",
#     "\n===== count =====",
#     f"2. num_task_name: {len(unique_task_names)}",
#     f"3. all:",
#     f"   - all_num_episode: {total_episodes}",
#     f"   - all_num_skill: {total_skills}",
#     "\n===== detail =====",
#     "4. num_episode/task num_skill/task:"
# ]

# # 添加每个task的详细统计（包括来源文件）
# detailed_stats = []
# for task_name, stats in task_stats.items():
#     detailed_stats.append({
#         "task_name": task_name,
#         "episode_count": stats["episodes"],
#         "skill_count": stats["skills"],
#         "source_files": sorted(list(stats["source_files"]))  # 转为有序列表
#     })

# output_content.append(json.dumps(detailed_stats, indent=2, ensure_ascii=False))

# # 同时输出到控制台和文件
# with open("/fs-computility/efm/shared/datasets/agibot-world/AgiBotWorld-Alpha/count_agibot2.txt", "w", encoding="utf-8") as f:
#     for line in output_content:
#         # 输出到控制台
#         print(line)
#         # 写入文件（处理JSON字符串的特殊情况）
#         if isinstance(line, str) and line.startswith('['):
#             f.write(line + '\n')
#         else:
#             f.write(f"{line}\n")

import json
import os
from collections import defaultdict

# 初始化统计字典
task_stats = defaultdict(lambda: {
    "episodes": 0,
    "skills": 0,
    "source_files": defaultdict(list),  # 结构：{文件名: [init_scene_text]}
    "unique_scenes": set()
})
task_files_count = 0
unique_task_names = set()

# 遍历task_info目录下的所有JSON文件
task_info_dir = "/fs-computility/efm/shared/datasets/agibot-world/AgiBotWorld-Alpha/main/task_info"
for filename in os.listdir(task_info_dir):
    if filename.startswith("task_") and filename.endswith(".json"):
        task_files_count += 1
        filepath = os.path.join(task_info_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                episodes = json.load(f)
                for episode in episodes:
                    task_name = episode["task_name"]
                    scene_text = episode["init_scene_text"]
                    
                    # 更新统计信息
                    unique_task_names.add(task_name)
                    task_stats[task_name]["source_files"][filename].append(scene_text)
                    task_stats[task_name]["unique_scenes"].add(scene_text)
                    task_stats[task_name]["episodes"] += 1
                    task_stats[task_name]["skills"] += len(episode["label_info"]["action_config"])
            except json.JSONDecodeError:
                print(f"错误：文件 {filename} 格式无效")

# 构建输出内容
output = [
    "===== Directory Analysis =====",
    f"Number of task files: {task_files_count}",
    "\n===== Global Statistics =====",
    f"Unique task types: {len(unique_task_names)}",
    f"Total episodes: {sum(t['episodes'] for t in task_stats.values())}",
    f"Total skills: {sum(t['skills'] for t in task_stats.values())}",
    "\n===== Detailed Task Breakdown ====="
]

# 添加每个任务的详细统计
for task_name, data in task_stats.items():
    task_output = [
        f"\nTask: {task_name}",
        f"- Episodes: {data['episodes']}",
        f"- Skills: {data['skills']}",
        f"- Unique scene descriptions: {len(data['unique_scenes'])}",
        "Source files with scene texts:"
    ]
    
    # 添加每个文件的场景描述
    for file, scenes in data["source_files"].items():
        unique_in_file = len(set(scenes))
        task_output.append(f"  • {file}")
        task_output.append(f"    Different scenes: {unique_in_file}")
        for idx, text in enumerate(set(scenes), 1):
            task_output.append(f"      {idx}. {text}")
    
    output.extend(task_output)

# 写入文件
output_path = "/fs-computility/efm/shared/datasets/agibot-world/AgiBotWorld-Alpha/count_agibot2.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print(f"统计完成，结果已保存至 {output_path}")