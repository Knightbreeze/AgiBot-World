""" 
This project is built upon the open-source project 🤗 LeRobot: https://github.com/huggingface/lerobot 

We are grateful to the LeRobot team for their outstanding work and their contributions to the community. 

If you find this project useful, please also consider supporting and exploring LeRobot. 
"""

import os
import json
import shutil
import logging
import argparse
import gc
from pathlib import Path
from typing import Callable, Optional, List
from functools import partial
from math import ceil
from copy import deepcopy

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

# 保持原有的常量定义和函数不变
# [保持HEAD_COLOR到FEATURES的所有代码不变]
# [保持get_stats_einops_patterns到AgiBotDataset的所有代码不变]

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


HEAD_COLOR = "head_color.mp4"
HAND_LEFT_COLOR = "hand_left_color.mp4"
HAND_RIGHT_COLOR = "hand_right_color.mp4"
# HEAD_CENTER_FISHEYE_COLOR = "head_center_fisheye_color.mp4"
# HEAD_LEFT_FISHEYE_COLOR = "head_left_fisheye_color.mp4"
# HEAD_RIGHT_FISHEYE_COLOR = "head_right_fisheye_color.mp4"
# BACK_LEFT_FISHEYE_COLOR = "back_left_fisheye_color.mp4"
# BACK_RIGHT_FISHEYE_COLOR = "back_right_fisheye_color.mp4"
HEAD_DEPTH = "head_depth"

DEFAULT_IMAGE_PATH = (
    "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.jpg"
)

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
    # 移除了以下fisheye相机配置：
    # - observation.images.head_center_fisheye
    # - observation.images.head_left_fisheye
    # - observation.images.head_right_fisheye 
    # - observation.images.back_left_fisheye
    # - observation.images.back_right_fisheye
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

# 创建目录结构（在main函数开始前执行）
def setup_directories():
    """初始化共享存储目录结构"""
    required_dirs = [
        CACHE_ROOT / "huggingface/datasets",
        CACHE_ROOT / "torch",
        CACHE_ROOT / "tmp",
        CACHE_ROOT / "ffmpeg",
        CACHE_ROOT / "multiprocessing"
    ]
    for d in required_dirs:
        d.mkdir(parents=True, exist_ok=True)
        # 设置宽松权限（根据实际安全要求调整）
        os.chmod(d, 0o777)

def get_stats_einops_patterns(dataset, num_workers=0):
    """These einops patterns will be used to aggregate batches and compute statistics.

    Note: We assume the images are in channel first format
    """

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=4,
        shuffle=False,
    )
    batch = next(iter(dataloader))

    stats_patterns = {}

    for key in dataset.features:
        # sanity check that tensors are not float64
        assert batch[key].dtype != torch.float64

        # if isinstance(feats_type, (VideoFrame, Image)):
        if key in dataset.meta.camera_keys:
            # sanity check that images are channel first
            _, c, h, w = batch[key].shape
            assert (
                c < h and c < w
            ), f"expect channel first images, but instead {batch[key].shape}"
            assert (
                batch[key].dtype == torch.float32
            ), f"expect torch.float32, but instead {batch[key].dtype=}"
            # assert batch[key].max() <= 1, f"expect pixels lower than 1, but instead {batch[key].max()=}"
            # assert batch[key].min() >= 0, f"expect pixels greater than 1, but instead {batch[key].min()=}"
            stats_patterns[key] = "b c h w -> c 1 1"
        elif batch[key].ndim == 2:
            stats_patterns[key] = "b c -> c "
        elif batch[key].ndim == 1:
            stats_patterns[key] = "b -> 1"
        else:
            raise ValueError(f"{key}, {batch[key].shape}")

    return stats_patterns

def compute_stats(dataset, batch_size=64, num_workers=4, max_num_samples=None):
    """Compute mean/std and min/max statistics of all data keys in a LeRobotDataset."""
    if max_num_samples is None:
        max_num_samples = len(dataset)

    # for more info on why we need to set the same number of workers, see `load_from_videos`
    stats_patterns = get_stats_einops_patterns(dataset, num_workers)

    # mean and std will be computed incrementally while max and min will track the running value.
    mean, std, max, min = {}, {}, {}, {}
    for key in stats_patterns:
        mean[key] = torch.tensor(0.0).float()
        std[key] = torch.tensor(0.0).float()
        max[key] = torch.tensor(-float("inf")).float()
        min[key] = torch.tensor(float("inf")).float()

    def create_seeded_dataloader(dataset, batch_size, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        return dataloader

    # Note: Due to be refactored soon. The point of storing `first_batch` is to make sure we don't get
    # surprises when rerunning the sampler.
    first_batch = None
    running_item_count = 0  # for online mean computation
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
        tqdm(
            dataloader,
            total=ceil(max_num_samples / batch_size),
            desc="Compute mean, min, max",
        )
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        if first_batch is None:
            first_batch = deepcopy(batch)
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation.
            batch_mean = einops.reduce(batch[key], pattern, "mean")
            # Hint: to update the mean we need x̄ₙ = (Nₙ₋₁x̄ₙ₋₁ + Bₙxₙ) / Nₙ, where the subscript represents
            # the update step, N is the running item count, B is this batch size, x̄ is the running mean,
            # and x is the current batch mean. Some rearrangement is then required to avoid risking
            # numerical overflow. Another hint: Nₙ₋₁ = Nₙ - Bₙ. Rearrangement yields
            # x̄ₙ = x̄ₙ₋₁ + Bₙ * (xₙ - x̄ₙ₋₁) / Nₙ
            mean[key] = (
                mean[key]
                + this_batch_size * (batch_mean - mean[key]) / running_item_count
            )
            max[key] = torch.maximum(
                max[key], einops.reduce(batch[key], pattern, "max")
            )
            min[key] = torch.minimum(
                min[key], einops.reduce(batch[key], pattern, "min")
            )

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    first_batch_ = None
    running_item_count = 0  # for online std computation
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
        tqdm(dataloader, total=ceil(max_num_samples / batch_size), desc="Compute std")
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        # Sanity check to make sure the batches are still in the same order as before.
        if first_batch_ is None:
            first_batch_ = deepcopy(batch)
            for key in stats_patterns:
                assert torch.equal(first_batch_[key], first_batch[key])
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation (where the mean is over squared
            # residuals).See notes in the mean computation loop above.
            batch_std = einops.reduce((batch[key] - mean[key]) ** 2, pattern, "mean")
            std[key] = (
                std[key] + this_batch_size * (batch_std - std[key]) / running_item_count
            )

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    for key in stats_patterns:
        std[key] = torch.sqrt(std[key])

    stats = {}
    for key in stats_patterns:
        stats[key] = {
            "mean": mean[key],
            "std": std[key],
            "max": max[key],
            "min": min[key],
        }
    return stats

class AgiBotDataset(LeRobotDataset):
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
        We rewrite this method to copy mp4 videos to the target position
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        episode_length = episode_buffer.pop("size")
        episode_index = episode_buffer["episode_index"]
        if episode_index != self.meta.total_episodes:
            # TODO(aliberts): Add option to use existing episode_index
            raise NotImplementedError(
                "You might have manually provided the episode_buffer with an episode_index that doesn't "
                "match the total number of episodes in the dataset. This is not supported for now."
            )

        if episode_length == 0:
            raise ValueError(
                "You must add one or several frames with `add_frame` before calling `add_episode`."
            )

        task_index = self.meta.get_task_index(task)

        if not set(episode_buffer.keys()) == set(self.features):
            raise ValueError()

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
            else:
                raise ValueError(key)

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)

        self.meta.save_episode(episode_index, episode_length, task, task_index)
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            episode_buffer[key] = video_path
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(videos[key], video_path)
        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()
        self.consolidated = False

    def consolidate(
        self, run_compute_stats: bool = True, keep_image_files: bool = False
    ) -> None:
        self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(
            self.meta.episodes, self.episodes
        )
        check_timestamps_sync(
            self.hf_dataset, self.episode_data_index, self.fps, self.tolerance_s
        )
        if len(self.meta.video_keys) > 0:
            self.meta.write_video_info()

        if not keep_image_files:
            img_dir = self.root / "images"
            if img_dir.is_dir():
                shutil.rmtree(self.root / "images")
        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        if run_compute_stats:
            self.stop_image_writer()
            self.meta.stats = compute_stats(self)
            serialized_stats = serialize_dict(self.meta.stats)
            write_json(serialized_stats, self.root / STATS_PATH)
            self.consolidated = True
        else:
            logging.warning(
                "Skipping computation of the dataset statistics, dataset is not fully consolidated."
            )

    def add_frame(self, frame: dict) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # TODO(aliberts, rcadene): Add sanity check for the input, check it's numpy or torch,
        # check the dtype and shape matches, etc.

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        frame_index = self.episode_buffer["size"]
        timestamp = (
            frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        )
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        for key in frame:
            if key not in self.features:
                raise ValueError(key)
            item = (
                frame[key].numpy()
                if isinstance(frame[key], torch.Tensor)
                else frame[key]
            )
            self.episode_buffer[key].append(item)

        self.episode_buffer["size"] += 1

def load_depths(root_dir: str, camera_name: str):
    cam_path = Path(root_dir)
    all_imgs = sorted(list(cam_path.glob(f"{camera_name}*")))
    return [np.array(Image.open(f)).astype(np.float32) / 1000 for f in all_imgs]

def load_local_dataset(episode_id: int, src_path: str, task_id: int) -> Optional[tuple]:
    """Load local dataset and return a dict with observations and actions"""
    try:
        ob_dir = Path(src_path) / f"observations/{task_id}/{episode_id}"
        depth_imgs = load_depths(ob_dir / "depth", HEAD_DEPTH)
        proprio_dir = Path(src_path) / f"proprio_stats/{task_id}/{episode_id}"

        with h5py.File(proprio_dir / "proprio_stats.h5") as f:
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
            logging.warning(f"Data length mismatch in episode {episode_id}")
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
            # 移除了以下fisheye视频路径：
            # "observation.images.head_center_fisheye": v_path / HEAD_CENTER_FISHEYE_COLOR,
            # "observation.images.head_left_fisheye": v_path / HEAD_LEFT_FISHEYE_COLOR,
            # "observation.images.head_right_fisheye": v_path / HEAD_RIGHT_FISHEYE_COLOR,
            # "observation.images.back_left_fisheye": v_path / BACK_LEFT_FISHEYE_COLOR,
            # "observation.images.back_right_fisheye": v_path / BACK_RIGHT_FISHEYE_COLOR,
        }
        return (frames, videos)
    except Exception as e:
        logging.error(f"Error loading episode {episode_id}: {str(e)}")
        return None

def get_task_instruction(task_json_path: str) -> str:
    """Get task language instruction with validation"""
    try:
        with open(task_json_path, "r") as f:
            task_info = json.load(f)
        if not task_info or not isinstance(task_info, list):
            raise ValueError("Invalid task info format")
            
        task_name = task_info[0].get("task_name", "")
        task_init_scene = task_info[0].get("init_scene_text", "")
        return f"{task_name}.{task_init_scene}"
    except Exception as e:
        logging.error(f"Error loading task instruction: {str(e)}")
        return "unknown_task"

def process_single_task(
    task_id: int,
    src_path: str,
    dataset: AgiBotDataset,
    debug: bool,
    chunk_size: int
):
    """处理单个任务的所有episodes"""
    task_json = Path(src_path) / "task_info" / f"task_{task_id}.json"
    if not task_json.exists():
        logging.warning(f"Task {task_id} JSON not found, skipping.")
        return

    task_instruction = get_task_instruction(str(task_json))
    logging.info(f"Processing task {task_id}: {task_instruction[:50]}...")

    # 获取所有episode目录
    episodes_dir = Path(src_path) / "observations" / str(task_id)
    if not episodes_dir.exists():
        logging.warning(f"No observations directory for task {task_id}")
        return
        
    episode_ids = [d.name for d in episodes_dir.iterdir() if d.is_dir()]
    try:
        episode_ids = sorted(map(int, episode_ids))
    except ValueError:
        logging.warning(f"Invalid episode IDs in {episodes_dir}")
        return

    if debug:
        episode_ids = episode_ids[:2]

    # 处理每个episode
    for chunk_start in tqdm(range(0, len(episode_ids), chunk_size), 
                          desc=f"Processing task {task_id}"):
        chunk_end = min(chunk_start + chunk_size, len(episode_ids))
        chunk_ids = episode_ids[chunk_start:chunk_end]

        # 加载数据
        if debug:
            episodes_data = [
                load_local_dataset(eid, src_path, task_id)
                for eid in chunk_ids
            ]
        else:
            episodes_data = process_map(
                partial(load_local_dataset, src_path=src_path, task_id=task_id),
                chunk_ids,
                max_workers=os.cpu_count(),  # 使用所有可用CPU核心
                desc=f"Loading chunk {chunk_start//chunk_size + 1}"
            )

        # 处理有效数据
        valid_episodes = [ep for ep in episodes_data if ep is not None]
        for episode_data in valid_episodes:
            frames, videos = episode_data
            for frame in frames:
                dataset.add_frame(frame)
            dataset.save_episode(task=task_instruction, videos=videos)
        
        # 内存管理
        del episodes_data, valid_episodes
        gc.collect()

def main(
    src_path: str,
    tgt_path: str,
    repo_id: str,
    debug: bool = False,
    chunk_size: int = 10
):
    # 初始化共享存储目录
    setup_directories()

    # 强制设置huggingface配置
    from datasets import config
    config.HF_DATASETS_CACHE = os.environ["HF_DATASETS_CACHE"]
    
    """主转换函数"""
    # 自动发现所有task_id
    task_info_dir = Path(src_path) / "task_info"
    task_files = list(task_info_dir.glob("task_*.json"))
    if not task_files:
        raise ValueError("No task files found in task_info directory")
    
    task_ids = sorted([int(f.stem.split("_")[1]) for f in task_files])
    logging.info(f"Found {len(task_ids)} tasks: {task_ids}")

    # 初始化数据集
    dataset = AgiBotDataset.create(
        repo_id=repo_id,
        root=Path(tgt_path) / repo_id,
        fps=30,
        robot_type="a2d",
        features=FEATURES,
    )

    # 处理每个任务
    for task_id in task_ids:
        process_single_task(
            task_id=task_id,
            src_path=src_path,
            dataset=dataset,
            debug=debug,
            chunk_size=chunk_size
        )

    # 最终整合数据集
    logging.info("Consolidating final dataset...")
    dataset.consolidate()
    logging.info(f"Conversion completed. Dataset saved to: {Path(tgt_path)/repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_path",
        type=str,
        required=True,
        help="Path to source data directory"
    )
    parser.add_argument(
        "--tgt_path", 
        type=str,
        required=True,
        help="Output directory for converted dataset"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="agibotworld/all_tasks",
        help="HF repository ID for the dataset"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (process 2 episodes per task)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="Number of episodes to process at once"
    )

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

    args = parser.parse_args()
    
    # 验证源路径
    if not Path(args.src_path).exists():
        raise ValueError(f"Source path {args.src_path} does not exist")
        
    main(
        src_path=args.src_path,
        tgt_path=args.tgt_path,
        repo_id=args.repo_id,
        debug=args.debug,
        chunk_size=args.chunk_size
    )