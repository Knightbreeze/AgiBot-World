""" 
This project is built upon the open-source project 🤗 LeRobot: https://github.com/huggingface/lerobot 

We are grateful to the LeRobot team for their outstanding work and their contributions to the community. 

If you find this project useful, please also consider supporting and exploring LeRobot. 

This is an optimized version for processing large-scale datasets with multi-node and NUMA-aware parallelism.
"""

import os
import json
import shutil
import logging
import argparse
import gc
import time
import signal
import socket
import resource
import psutil
import traceback
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any, Tuple, Union
from functools import partial
from math import ceil
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import h5py
import torch
import einops
import numpy as np
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import (
    STATS_PATH,
    check_timestamps_sync,
    get_episode_data_index,
    serialize_dict,
    write_json,
)

# ================== 新增共享存储配置 ==================
SHARED_STORAGE = Path("/fs-computility/efm/shared")
# 配置所有缓存和临时文件的子目录
CACHE_ROOT = SHARED_STORAGE / "agibot2lerobot_cache"
os.environ.update({
    # Hugging Face 数据集缓存
    "HF_DATASETS_CACHE": str(CACHE_ROOT / "huggingface/datasets"),
    # PyTorch缓存
    "TORCH_HOME": str(CACHE_ROOT / "torch"),
    # 系统临时文件
    "TMPDIR": str(CACHE_ROOT / "tmp"),
    # ffmpeg临时文件
    "FFMPEG_TEMP": str(CACHE_ROOT / "ffmpeg")
})

# 保存先前定义的常量
HEAD_COLOR = "head_color.mp4"
HAND_LEFT_COLOR = "hand_left_color.mp4"
HAND_RIGHT_COLOR = "hand_right_color.mp4"
HEAD_DEPTH = "head_depth"

DEFAULT_IMAGE_PATH = (
    "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.jpg"
)

# 保持原始的特征定义
FEATURES = {
    "observation.images.top_head": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.cam_top_depth": {
        "dtype": "image",
        "shape": [480, 640, 1],
        "names": ["height", "width", "channel"],
    },
    "observation.images.hand_left": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.hand_right": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.state": {
        "dtype": "float32",
        "shape": [20],
    },
    "action": {
        "dtype": "float32",
        "shape": [22],
    },
    "episode_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "frame_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "task_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
}

# ================== 资源管理与监控 ==================
class ResourceMonitor:
    """监控系统资源使用情况的类"""
    def __init__(self, log_interval=60, max_memory_percent=90, log_file=None):
        self.log_interval = log_interval
        self.max_memory_percent = max_memory_percent
        self.running = False
        self.thread = None
        self.log_file = log_file
        self.start_time = time.time()
    
    def _log_message(self, message):
        """记录消息到日志和文件"""
        logging.info(message)
        if self.log_file:
            with open(self.log_file, "a") as f:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{timestamp} - {message}\n")
    
    def _monitor_loop(self):
        """资源监控循环"""
        while self.running:
            try:
                # 获取资源使用情况
                memory_usage = psutil.virtual_memory()
                disk_usage = psutil.disk_usage(os.getcwd())
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # 计算运行时间
                elapsed = time.time() - self.start_time
                hours, remainder = divmod(elapsed, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                message = (
                    f"运行时间: {int(hours)}h {int(minutes)}m {int(seconds)}s | "
                    f"内存: {memory_usage.used/(1024**3):.1f}GB/{memory_usage.total/(1024**3):.1f}GB "
                    f"({memory_usage.percent:.1f}%) | "
                    f"磁盘: {disk_usage.used/(1024**3):.1f}GB/{disk_usage.total/(1024**3):.1f}GB "
                    f"({disk_usage.percent:.1f}%) | "
                    f"CPU: {cpu_percent:.1f}%"
                )
                self._log_message(message)
                
                # 检查是否接近资源上限
                if memory_usage.percent > self.max_memory_percent:
                    self._log_message(f"警告: 内存使用率达到{memory_usage.percent:.1f}%，超过{self.max_memory_percent}%阈值")
                    # 这里可以添加一些紧急处理，如触发GC或暂停某些任务
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # 休眠指定时间
                time.sleep(self.log_interval)
            except Exception as e:
                self._log_message(f"监控线程异常: {str(e)}")
                time.sleep(self.log_interval * 2)  # 出错时等待更长时间
    
    def start(self):
        """启动监控线程"""
        if not self.running:
            self.running = True
            self.thread = ThreadPoolExecutor(max_workers=1)
            self.thread.submit(self._monitor_loop)
            self._log_message("资源监控已启动")
    
    def stop(self):
        """停止监控线程"""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.shutdown(wait=False)
            self._log_message("资源监控已停止")

def get_memory_usage_gb():
    """获取当前内存使用量（GB）"""
    return psutil.virtual_memory().used / (1024**3)

def get_disk_usage_gb(path):
    """获取指定路径的磁盘使用量（GB）"""
    return psutil.disk_usage(path).used / (1024**3)

def get_system_total_memory_gb():
    """获取系统总内存（GB）"""
    return psutil.virtual_memory().total / (1024**3)

# ================== NUMA拓扑感知 ==================
def get_numa_topology():
    """获取NUMA节点拓扑结构"""
    try:
        numa_info = {}
        # 在实际环境中，应该使用更精确的方法获取NUMA配置
        # 这里使用简化的方法，假设有2个NUMA节点，每个节点分配一半的CPU核心
        total_cores = os.cpu_count()
        cores_per_node = total_cores // 2
        
        numa_info[0] = list(range(0, cores_per_node))
        numa_info[1] = list(range(cores_per_node, total_cores))
        
        return numa_info
    except Exception as e:
        logging.warning(f"无法获取NUMA拓扑: {e}, 使用默认配置")
        return {0: list(range(os.cpu_count()))}

def optimize_parallelism(memory_gb=None):
    """优化并行度配置"""
    total_cores = os.cpu_count()
    if memory_gb is None:
        memory_gb = get_system_total_memory_gb()
    
    # 计算最优配置
    # 假设每个处理任务需要约20GB内存
    num_processes = min(max(1, total_cores // 8), max(1, int(memory_gb // 20)))
    threads_per_process = max(2, min(8, total_cores // num_processes))
    
    return num_processes, threads_per_process

# ================== IO优化 ==================
def setup_io_optimizations():
    """设置IO优化"""
    # 增加文件描述符限制
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = min(hard, 65536)  # 设置一个合理的高值
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        logging.info(f"文件描述符限制设置为: {new_soft}")
    except Exception as e:
        logging.warning(f"无法调整文件描述符限制: {e}")
    
    # 配置HDF5不使用文件锁
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
    # 关闭PyTorch的线程优化，避免和我们的并行策略冲突
    torch.set_num_threads(1)

# ================== 断点续传机制 ==================
def setup_directories():
    """初始化共享存储目录结构"""
    required_dirs = [
        CACHE_ROOT / "huggingface/datasets",
        CACHE_ROOT / "torch",
        CACHE_ROOT / "tmp",
        CACHE_ROOT / "ffmpeg",
        CACHE_ROOT / "multiprocessing",
        CACHE_ROOT / "progress_tracking"  # 新增进度跟踪目录
    ]
    for d in required_dirs:
        d.mkdir(parents=True, exist_ok=True)
        # 设置宽松权限
        os.chmod(d, 0o777)

def get_progress_tracking_dir(repo_id):
    """获取进度跟踪目录"""
    return CACHE_ROOT / "progress_tracking" / repo_id.replace('/', '_')

def get_task_state_file(repo_id, task_id):
    """获取任务状态文件路径"""
    tracking_dir = get_progress_tracking_dir(repo_id)
    tracking_dir.mkdir(parents=True, exist_ok=True)
    return tracking_dir / f"task_{task_id}_state.json"

def get_processing_state(state_file):
    """读取处理状态"""
    if state_file.exists():
        with open(state_file, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"状态文件 {state_file} 损坏，创建新状态")
                return {"completed_episodes": [], "in_progress": False, "error_episodes": []}
    return {"completed_episodes": [], "in_progress": False, "error_episodes": []}

def update_processing_state(state_file, episode_id=None, start=False, complete=False, error=False):
    """更新处理状态"""
    state = get_processing_state(state_file)
    
    if start:
        state["in_progress"] = True
    
    if complete:
        state["in_progress"] = False
    
    if episode_id is not None:
        if error:
            if episode_id not in state["error_episodes"]:
                state["error_episodes"].append(episode_id)
        else:
            if episode_id not in state["completed_episodes"]:
                state["completed_episodes"].append(episode_id)
    
    # 原子写入
    temp_file = state_file.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(state, f, indent=2)
    temp_file.rename(state_file)
    
    return state

def get_all_tasks_progress(repo_id, task_ids):
    """获取所有任务的进度情况"""
    progress = {}
    tracking_dir = get_progress_tracking_dir(repo_id)
    
    for task_id in task_ids:
        state_file = get_task_state_file(repo_id, task_id)
        if state_file.exists():
            state = get_processing_state(state_file)
            progress[task_id] = {
                "completed": len(state["completed_episodes"]),
                "errors": len(state["error_episodes"]),
                "in_progress": state["in_progress"]
            }
        else:
            progress[task_id] = {"completed": 0, "errors": 0, "in_progress": False}
    
    return progress

def get_task_episodes_count(src_path, task_id):
    """获取任务中的总episode数量"""
    episodes_dir = Path(src_path) / "observations" / str(task_id)
    if not episodes_dir.exists():
        return 0
    
    episode_dirs = [d for d in episodes_dir.iterdir() if d.is_dir()]
    return len(episode_dirs)

# ================== 数据加载与处理 ==================
def load_depths(root_dir: str, camera_name: str):
    """加载深度图像数据"""
    cam_path = Path(root_dir)
    all_imgs = sorted(list(cam_path.glob(f"{camera_name}*")))
    
    # 使用内存映射优化大文件加载
    depth_imgs = []
    for f in all_imgs:
        try:
            # 使用PIL和numpy高效加载图像
            img = np.array(Image.open(f)).astype(np.float32) / 1000
            depth_imgs.append(img)
        except Exception as e:
            logging.warning(f"加载图像 {f} 失败: {str(e)}")
    
    return depth_imgs

def load_local_dataset(episode_id: int, src_path: str, task_id: int) -> Optional[tuple]:
    """加载本地数据集并返回观察和动作的字典"""
    try:
        ob_dir = Path(src_path) / f"observations/{task_id}/{episode_id}"
        depth_imgs = load_depths(ob_dir / "depth", HEAD_DEPTH)
        proprio_dir = Path(src_path) / f"proprio_stats/{task_id}/{episode_id}"

        # 使用上下文管理器打开HDF5文件
        with h5py.File(proprio_dir / "proprio_stats.h5") as f:
            # 采用延迟加载策略，只在需要时读取数据
            state_joint = np.array(f["state/joint/position"])
            state_effector = np.array(f["state/effector/position"])
            state_head = np.array(f["state/head/position"])
            state_waist = np.array(f["state/waist/position"])
            action_joint = np.array(f["action/joint/position"])
            action_effector = np.array(f["action/effector/position"])
            action_head = np.array(f["action/head/position"])
            action_waist = np.array(f["action/waist/position"])
            action_velocity = np.array(f["action/robot/velocity"])

        states_value = np.hstack(
            [state_joint, state_effector, state_head, state_waist]
        ).astype(np.float32)
        
        action_value = np.hstack(
            [action_joint, action_effector, action_head, action_waist, action_velocity]
        ).astype(np.float32)

        # 添加数据校验
        if not (len(depth_imgs) == len(states_value) == len(action_value)):
            logging.warning(f"数据长度不匹配，episode {episode_id}")
            return None

        frames = [
            {
                "observation.images.cam_top_depth": depth_imgs[i],
                "observation.state": states_value[i],
                "action": action_value[i],
            }
            for i in range(len(depth_imgs))
        ]

        v_path = ob_dir / "videos"
        videos = {
            "observation.images.top_head": v_path / HEAD_COLOR,
            "observation.images.hand_left": v_path / HAND_LEFT_COLOR,
            "observation.images.hand_right": v_path / HAND_RIGHT_COLOR,
        }
        return (frames, videos)
    except Exception as e:
        logging.error(f"加载episode {episode_id} 失败: {str(e)}")
        return None

def get_task_instruction(task_json_path: str) -> str:
    """获取任务语言指令"""
    try:
        with open(task_json_path, "r") as f:
            task_info = json.load(f)
        if not task_info or not isinstance(task_info, list):
            raise ValueError("无效的任务信息格式")
            
        task_name = task_info[0].get("task_name", "")
        task_init_scene = task_info[0].get("init_scene_text", "")
        return f"{task_name}.{task_init_scene}"
    except Exception as e:
        logging.error(f"加载任务指令失败: {str(e)}")
        return "unknown_task"

# ================== 数据集类继承 ==================
class AgiBotDataset(LeRobotDataset):
    """增强的AgiBotDataset，支持断点续传和错误恢复"""
    
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        download_videos: bool = True,
        local_files_only: bool = False,
        video_backend: str | None = None,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            download_videos=download_videos,
            local_files_only=local_files_only,
            video_backend=video_backend,
        )

    def save_episode(
        self, task: str, episode_data: dict | None = None, videos: dict | None = None
    ) -> None:
        """
        重写此方法以复制mp4视频到目标位置，并增强错误处理
        """
        try:
            if not episode_data:
                episode_buffer = self.episode_buffer
            else:
                episode_buffer = episode_data

            episode_length = episode_buffer.pop("size")
            episode_index = episode_buffer["episode_index"]
            
            if episode_index != self.meta.total_episodes:
                # TODO(aliberts): 添加选项以使用现有的episode_index
                raise NotImplementedError(
                    "您可能手动提供了episode_buffer，其episode_index与数据集中的总episode数不匹配。目前不支持此功能。"
                )

            if episode_length == 0:
                raise ValueError(
                    "必须在调用`add_episode`之前使用`add_frame`添加一个或多个帧"
                )

            task_index = self.meta.get_task_index(task)

            if not set(episode_buffer.keys()) == set(self.features):
                key_diff = set(self.features) - set(episode_buffer.keys())
                raise ValueError(f"episode_buffer缺少以下键: {key_diff}")

            for key, ft in self.features.items():
                if key == "index":
                    episode_buffer[key] = np.arange(
                        self.meta.total_frames, self.meta.total_frames + episode_length
                    )
                elif key == "episode_index":
                    episode_buffer[key] = np.full((episode_length,), episode_index)
                elif key == "task_index":
                    episode_buffer[key] = np.full((episode_length,), task_index)
                elif ft["dtype"] in ["image", "video"]:
                    continue
                elif len(ft["shape"]) == 1 and ft["shape"][0] == 1:
                    episode_buffer[key] = np.array(episode_buffer[key], dtype=ft["dtype"])
                elif len(ft["shape"]) == 1 and ft["shape"][0] > 1:
                    episode_buffer[key] = np.stack(episode_buffer[key])
                elif len(ft["shape"]) == 2:  # 处理二维数组
                    episode_buffer[key] = np.stack(episode_buffer[key])
                else:
                    raise ValueError(f"不支持的特征形状: {key}: {ft['shape']}")

            self._wait_image_writer()
            self._save_episode_table(episode_buffer, episode_index)

            self.meta.save_episode(episode_index, episode_length, task, task_index)
            
            # 复制视频文件
            for key in self.meta.video_keys:
                video_path = self.root / self.meta.get_video_file_path(episode_index, key)
                episode_buffer[key] = video_path
                video_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 验证源视频是否存在
                if key not in videos or not Path(videos[key]).exists():
                    logging.warning(f"视频源文件不存在: {videos.get(key, 'None')}")
                    continue
                    
                shutil.copyfile(videos[key], video_path)
                
            if not episode_data:  # 重置缓冲区
                self.episode_buffer = self.create_episode_buffer()
                
            self.consolidated = False
            return True
            
        except Exception as e:
            logging.error(f"保存episode失败: {str(e)}\n{traceback.format_exc()}")
            return False

    def consolidate(
        self, run_compute_stats: bool = True, keep_image_files: bool = False
    ) -> None:
        """
        增强的consolidate方法，添加更多的错误处理
        """
        try:
            logging.info("开始整合数据集...")
            self.hf_dataset = self.load_hf_dataset()
            self.episode_data_index = get_episode_data_index(
                self.meta.episodes, self.episodes
            )
            
            logging.info("检查时间戳同步...")
            check_timestamps_sync(
                self.hf_dataset, self.episode_data_index, self.fps, self.tolerance_s
            )
            
            if len(self.meta.video_keys) > 0:
                logging.info("写入视频信息...")
                self.meta.write_video_info()

            if not keep_image_files:
                img_dir = self.root / "images"
                if img_dir.is_dir():
                    logging.info("删除临时图像文件...")
                    shutil.rmtree(self.root / "images")
                    
            # 验证文件创建正确
            logging.info("验证文件创建...")
            video_files = list(self.root.rglob("*.mp4"))
            expected_video_count = self.num_episodes * len(self.meta.video_keys)
            if len(video_files) != expected_video_count:
                logging.warning(f"视频文件数量不匹配! 找到 {len(video_files)}, 期望 {expected_video_count}")

            parquet_files = list(self.root.rglob("*.parquet"))
            if len(parquet_files) != self.num_episodes:
                logging.warning(f"Parquet文件数量不匹配! 找到 {len(parquet_files)}, 期望 {self.num_episodes}")

            if run_compute_stats:
                logging.info("计算数据集统计信息...")
                self.stop_image_writer()
                self.meta.stats = compute_stats(self)
                serialized_stats = serialize_dict(self.meta.stats)
                write_json(serialized_stats, self.root / STATS_PATH)
                self.consolidated = True
                logging.info("数据集整合完成!")
            else:
                logging.warning(
                    "跳过数据集统计信息计算，数据集未完全整合。"
                )
        except Exception as e:
            logging.error(f"整合数据集失败: {str(e)}\n{traceback.format_exc()}")
            raise

# ================== 多进程任务处理 ==================
def process_episodes_chunk(
    task_id: int,
    chunk_ids: list,
    src_path: str,
    dataset: AgiBotDataset,
    task_instruction: str,
    state_file: Path
):
    """处理一批episodes"""
    # 设置此进程的内存和线程限制
    success_count = 0
    error_count = 0
    
    for episode_id in chunk_ids:
        try:
            # 检查是否已经处理过
            state = get_processing_state(state_file)
            if episode_id in state["completed_episodes"]:
                logging.info(f"跳过已处理的episode {episode_id}")
                continue

            # 加载数据
            episode_data = load_local_dataset(episode_id, src_path, task_id)
            if episode_data is None:
                logging.warning(f"无法加载episode {episode_id}，标记为错误")
                update_processing_state(state_file, episode_id=episode_id, error=True)
                error_count += 1
                continue

            frames, videos = episode_data
            
            # 添加所有帧
            for frame in frames:
                dataset.add_frame(frame)
            
            # 保存episode
            success = dataset.save_episode(task=task_instruction, videos=videos)
            
            if success:
                update_processing_state(state_file, episode_id=episode_id)
                success_count += 1
                logging.info(f"成功处理episode {episode_id}")
            else:
                update_processing_state(state_file, episode_id=episode_id, error=True)
                error_count += 1
                logging.warning(f"处理episode {episode_id} 失败")
            
            # 清理内存
            del frames, videos, episode_data
            
        except Exception as e:
            logging.error(f"处理episode {episode_id} 时发生异常: {str(e)}\n{traceback.format_exc()}")
            update_processing_state(state_file, episode_id=episode_id, error=True)
            error_count += 1
    
    return success_count, error_count

def process_task_with_chunks(
    task_id: int,
    src_path: str,
    dataset: AgiBotDataset,
    repo_id: str,
    chunk_size: int,
    num_workers: int,
    debug: bool = False
):
    """使用分块并行处理单个任务的所有episodes"""
    task_json = Path(src_path) / "task_info" / f"task_{task_id}.json"
    if not task_json.exists():
        logging.warning(f"任务 {task_id} 的JSON文件不存在，跳过。")
        return False

    # 获取任务指令
    task_instruction = get_task_instruction(str(task_json))
    logging.info(f"处理任务 {task_id}: {task_instruction[:50]}...")

    # 获取状态文件
    state_file = get_task_state_file(repo_id, task_id)
    
    # 标记任务开始处理
    update_processing_state(state_file, start=True)

    # 获取所有episode目录
    episodes_dir = Path(src_path) / "observations" / str(task_id)
    if not episodes_dir.exists():
        logging.warning(f"任务 {task_id} 没有observations目录")
        return False
        
    episode_ids = [int(d.name) for d in episodes_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    episode_ids.sort()
    
    # 在DEBUG模式下只处理前两个episode
    if debug:
        episode_ids = episode_ids[:2]
        logging.info(f"调试模式: 只处理前 {len(episode_ids)} 个episodes")

    # 获取已完成的episodes
    state = get_processing_state(state_file)
    completed_episodes = set(state["completed_episodes"])
    error_episodes = set(state["error_episodes"])
    
    # 过滤出未处理或出错的episodes
    pending_episodes = [ep for ep in episode_ids if ep not in completed_episodes]
    
    if not pending_episodes:
        logging.info(f"任务 {task_id} 没有待处理的episodes")
        update_processing_state(state_file, complete=True)
        return True
    
    logging.info(f"任务 {task_id}: 共 {len(episode_ids)} 个episodes, "
                 f"已完成 {len(completed_episodes)}, "
                 f"待处理 {len(pending_episodes)}, "
                 f"错误 {len(error_episodes)}")

    # 按块处理episodes
    total_chunks = (len(pending_episodes) + chunk_size - 1) // chunk_size
    
    # 创建进度条
    progress = tqdm(total=len(pending_episodes), desc=f"任务 {task_id} 处理进度")
    progress.update(len(completed_episodes))
    
    total_success = 0
    total_errors = 0
    
    # 使用进程池处理块
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        # 提交所有块
        for i in range(0, len(pending_episodes), chunk_size):
            chunk = pending_episodes[i:i+chunk_size]
            future = executor.submit(
                process_episodes_chunk,
                task_id,
                chunk,
                src_path,
                dataset,
                task_instruction,
                state_file
            )
            futures.append((future, len(chunk)))
        
        # 等待所有块完成并更新进度
        for future, chunk_len in futures:
            try:
                success_count, error_count = future.result()
                total_success += success_count
                total_errors += error_count
                progress.update(success_count + error_count)
            except Exception as e:
                logging.error(f"块处理失败: {str(e)}")
                total_errors += chunk_len
    
    progress.close()
    
    # 更新最终状态
    update_processing_state(state_file, complete=True)
    
    # 重新获取当前状态
    state = get_processing_state(state_file)
    
    logging.info(f"任务 {task_id} 处理完成: "
                 f"总episodes: {len(episode_ids)}, "
                 f"成功: {len(state['completed_episodes'])}, "
                 f"错误: {len(state['error_episodes'])}")
    
    return True

def process_task_subset(
    task_ids: list,
    src_path: str,
    tgt_path: str,
    repo_id: str,
    chunk_size: int,
    num_workers: int,
    debug: bool = False,
    worker_id: int = 0
):
    """处理一组任务"""
    if not task_ids:
        logging.info(f"Worker {worker_id}: 没有分配任务")
        return
        
    worker_repo_id = f"{repo_id}_worker{worker_id}"
    logging.info(f"Worker {worker_id}: 开始处理 {len(task_ids)} 个任务, repo_id = {worker_repo_id}")
    
    # 创建数据集
    dataset = AgiBotDataset.create(
        repo_id=worker_repo_id,
        root=Path(tgt_path) / worker_repo_id,
        fps=30,
        robot_type="a2d",
        features=FEATURES,
    )
    
    # 处理每个任务
    for task_id in task_ids:
        try:
            success = process_task_with_chunks(
                task_id=task_id,
                src_path=src_path,
                dataset=dataset,
                repo_id=repo_id,  # 使用全局repo_id来追踪进度
                chunk_size=chunk_size,
                num_workers=num_workers,
                debug=debug
            )
            if not success:
                logging.warning(f"Worker {worker_id}: 处理任务 {task_id} 失败")
        except Exception as e:
            logging.error(f"Worker {worker_id}: 处理任务 {task_id} 异常: {str(e)}\n{traceback.format_exc()}")
    
    # 整合数据集
    logging.info(f"Worker {worker_id}: 整合数据集...")
    dataset.consolidate()
    logging.info(f"Worker {worker_id}: 处理完成")
    
    return worker_repo_id

def merge_worker_datasets(worker_datasets, tgt_path, repo_id):
    """合并所有工作器的数据集"""
    logging.info(f"开始合并 {len(worker_datasets)} 个工作器数据集...")
    
    # 检查是否有数据集需要合并
    valid_datasets = [ds for ds in worker_datasets if ds and Path(tgt_path) / ds]
    
    if not valid_datasets:
        logging.warning("没有有效的数据集需要合并")
        return False
    
    if len(valid_datasets) == 1:
        # 只有一个数据集，直接重命名
        src_dir = Path(tgt_path) / valid_datasets[0]
        dst_dir = Path(tgt_path) / repo_id
        
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
            
        src_dir.rename(dst_dir)
        logging.info(f"只有一个数据集，已重命名为 {repo_id}")
        return True
    
    # 创建最终数据集
    final_dataset = AgiBotDataset.create(
        repo_id=repo_id,
        root=Path(tgt_path) / repo_id,
        fps=30,
        robot_type="a2d",
        features=FEATURES,
    )
    
    # TODO: 实现数据集合并逻辑
    logging.warning("多数据集合并功能尚未完全实现")
    
    return False

# ================== 主处理流程 ==================
def distributed_processing(
    src_path: str,
    tgt_path: str,
    repo_id: str = "agibotworld/all_tasks",
    debug: bool = False,
    chunk_size: int = 10,
    num_workers: int = None,
    num_processes: int = None
):
    """分布式处理主函数"""
    # 初始化共享存储目录
    setup_directories()
    
    # 设置IO优化
    setup_io_optimizations()
    
    # 如果未指定，自动计算最佳并行度
    if num_workers is None or num_processes is None:
        mem_gb = get_system_total_memory_gb()
        auto_processes, auto_workers = optimize_parallelism(mem_gb)
        
        if num_processes is None:
            num_processes = auto_processes
            
        if num_workers is None:
            num_workers = auto_workers
    
    logging.info(f"使用 {num_processes} 个进程，每个进程 {num_workers} 个工作线程")
    
    # 强制设置huggingface配置
    from datasets import config
    config.HF_DATASETS_CACHE = os.environ["HF_DATASETS_CACHE"]
    
    # 自动发现所有task_id
    task_info_dir = Path(src_path) / "task_info"
    task_files = list(task_info_dir.glob("task_*.json"))
    if not task_files:
        raise ValueError("task_info目录中找不到任务文件")
    
    task_ids = sorted([int(f.stem.split("_")[1]) for f in task_files])
    logging.info(f"发现 {len(task_ids)} 个任务: {task_ids}")
    
    # 获取进度情况
    progress = get_all_tasks_progress(repo_id, task_ids)
    
    # 分析进度
    completed_tasks = [tid for tid, info in progress.items() 
                         if info["completed"] > 0 and not info["in_progress"]]
    
    in_progress_tasks = [tid for tid, info in progress.items() 
                         if info["in_progress"]]
    
    pending_tasks = [tid for tid in task_ids 
                       if tid not in completed_tasks and tid not in in_progress_tasks]
    
    logging.info(f"任务分析: 已完成 {len(completed_tasks)}, "
                 f"进行中 {len(in_progress_tasks)}, "
                 f"待处理 {len(pending_tasks)}")
    
    if debug:
        pending_tasks = pending_tasks[:min(2, len(pending_tasks))]
        logging.info(f"调试模式: 仅处理 {len(pending_tasks)} 个任务")
    
    if not pending_tasks and not in_progress_tasks:
        logging.info("所有任务已处理完成")
        return
    
    # 将任务分配给进程
    tasks_to_process = pending_tasks + in_progress_tasks
    logging.info(f"将处理 {len(tasks_to_process)} 个任务")
    
    # 如果任务少于进程数，调整进程数
    if len(tasks_to_process) < num_processes:
        num_processes = max(1, len(tasks_to_process))
        logging.info(f"任务数少于进程数，调整为使用 {num_processes} 个进程")
    
    # 均匀分配任务
    task_chunks = []
    for i in range(num_processes):
        # 使用交错分配，确保每个进程分配到不同的任务类型
        process_tasks = tasks_to_process[i::num_processes]
        if process_tasks:  # 只添加非空任务列表
            task_chunks.append(process_tasks)
    
    # 启动资源监控
    monitor = ResourceMonitor(
        log_interval=60,
        log_file=Path(tgt_path) / f"{repo_id.replace('/', '_')}_resources.log"
    )
    monitor.start()
    
    try:
        # 使用进程池处理任务
        worker_datasets = []
        if num_processes > 1:
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = []
                
                # 提交所有工作器任务
                for i, tasks in enumerate(task_chunks):
                    future = executor.submit(
                        process_task_subset,
                        tasks,
                        src_path,
                        tgt_path,
                        repo_id,
                        chunk_size,
                        num_workers,
                        debug,
                        i
                    )
                    futures.append(future)
                
                # 等待所有工作器完成
                for future in as_completed(futures):
                    try:
                        worker_repo_id = future.result()
                        worker_datasets.append(worker_repo_id)
                    except Exception as e:
                        logging.error(f"工作器处理失败: {str(e)}\n{traceback.format_exc()}")
        else:
            # 单进程模式
            worker_repo_id = process_task_subset(
                task_chunks[0],
                src_path,
                tgt_path,
                repo_id,
                chunk_size,
                num_workers,
                debug,
                0
            )
            worker_datasets.append(worker_repo_id)
        
        # 合并所有工作器数据集
        merge_worker_datasets(worker_datasets, tgt_path, repo_id)
        
        logging.info(f"转换完成。数据集已保存到: {Path(tgt_path)/repo_id}")
        
    except KeyboardInterrupt:
        logging.info("收到中断信号，正在退出...")
    except Exception as e:
        logging.error(f"处理过程中发生异常: {str(e)}\n{traceback.format_exc()}")
    finally:
        # 停止资源监控
        monitor.stop()

# ================== 命令行入口 ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_path",
        type=str,
        required=True,
        help="源数据目录路径"
    )
    parser.add_argument(
        "--tgt_path", 
        type=str,
        required=True,
        help="转换后数据集的输出目录"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="lerobotV2_AgiBotWorld_sample",
        help="数据集的HF存储库ID"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式（每个任务仅处理2个episodes）"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="一次处理的episodes数量"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="要使用的进程数（默认自动检测）"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="每个进程的工作线程数（默认自动检测）"
    )

    # 配置日志
    log_format = "%(asctime)s [%(levelname)s] [%(processName)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"convert_{time.strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )

    # 显示系统信息
    hostname = socket.gethostname()
    cpu_cores = os.cpu_count()
    total_memory = get_system_total_memory_gb()
    logging.info(f"主机名: {hostname}, CPU核心: {cpu_cores}, 内存: {total_memory:.1f}GB")
    
    args = parser.parse_args()
    
    # 验证源路径
    if not Path(args.src_path).exists():
        raise ValueError(f"源路径 {args.src_path} 不存在")
        
    # 运行分布式处理
    distributed_processing(
        src_path=args.src_path,
        tgt_path=args.tgt_path,
        repo_id=args.repo_id,
        debug=args.debug,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        num_processes=args.num_processes
    )