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
import cv2

from scipy.spatial.transform import Rotation
import lmdb
import pickle
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


FEATURES = {
    "observation.images.Primary_0_0": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 15.0,  # 修改帧率为15
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.Wrist_0_0": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 15.0,  # 修改帧率为15
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.robot.qpos": {
        "dtype": "float32",
        "shape": [7],  # 假设关节位置维度为7
    },
    "observation.robot.ee_pose": {
        "dtype": "float32",
        "shape": [6],  # 4x4 -> 6D位姿
    },
    "action": {
        "dtype": "float32",
        "shape": [7],  # 6D位姿 + 夹爪状态
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


def pose_to_6d(pose, degrees=False):
    pose6d = np.zeros(6)
    pose6d[:3] = pose[:3, 3]  # 提取位置（平移向量）
    pose6d[3:6] = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=degrees)  # 提取旋转（欧拉角）
    return pose6d


def get_stats_einops_patterns(dataset, num_workers=0):
    """These einops patterns will be used to aggregate batches and compute statistics.

    Note: We assume the images are in channel first format
    """

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=2,
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


class BananaDataset(LeRobotDataset):
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
            elif len(ft["shape"]) == 2:  # 处理二维数组，如ee_pose (4,4)
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
        frame_index = self.episode_buffer["frame_index"][-1] + 1 if frame_index > 0 else 0
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

def load_lmdb_data(episode_path: Path, meta: dict):
    env = lmdb.open(str(episode_path/"lmdb"), 
                  readonly=True, 
                  lock=False,
                  max_readers=128,
                  readahead=False)
    
    frames = []
    with env.begin(write=False) as txn:
        language_instruction = "Grasp the brush and empty the objects on the chopping board into the dustpan."
        # 预加载所有键
        existing_keys = set(txn.cursor().iternext(values=False))
        
        # 确定图像数据的数量
        image_keys = [k for k in existing_keys if b'color_image' in k]
        primary_keys = sorted([k for k in image_keys if b'Primary_0_0' in k])
        if not primary_keys:
            logging.error(f"No image data found in {episode_path.name}")
            return None
        
        total_steps = len(primary_keys)
        
        # 加载全局数据（如果是作为单一序列存储的）
        try:
            # 检查这些键是否存在
            scalar_keys = [b'delta_arm_ee_action', b'gripper_action', 
                          b'observation/robot/qpos', b'observation/robot/forlan2robot_pose']
            
            if all(k in existing_keys for k in scalar_keys):
                # 所有标量数据作为序列存储
                all_delta_actions = pickle.loads(txn.get(b'delta_arm_ee_action'))
                all_gripper_actions = pickle.loads(txn.get(b'gripper_action'))
                all_qpos = pickle.loads(txn.get(b'observation/robot/qpos'))
                all_ee_poses = pickle.loads(txn.get(b'observation/robot/forlan2robot_pose'))
                
                # 验证数据长度
                if len(all_delta_actions) != total_steps or len(all_gripper_actions) != total_steps:
                    logging.warning(f"Action data length mismatch in {episode_path.name}")
                    return None
                    
                scalar_data_mode = "sequence"
            else:
                # 按步骤单独存储
                scalar_data_mode = "step"
        except Exception as e:
            logging.error(f"Error loading scalar data: {str(e)}")
            return None
            
        for step in range(total_steps):
            try:
                # 图像键使用4位数字格式
                step_str_4 = f"{step:04d}"
                
                # 构建图像键
                primary_img_key = f"observation/Primary_0_0/color_image/{step_str_4}".encode()
                wrist_img_key = f"observation/Wrist_0_0/color_image/{step_str_4}".encode()
                
                # 检查图像键是否存在
                if primary_img_key not in existing_keys or wrist_img_key not in existing_keys:
                    continue
                    
                # 读取图像数据
                primary_img = cv2.imdecode(
                    pickle.loads(txn.get(primary_img_key)),
                    cv2.IMREAD_COLOR)
                wrist_img = cv2.imdecode(
                    pickle.loads(txn.get(wrist_img_key)),
                    cv2.IMREAD_COLOR)                
                # 获取标量数据（根据存储模式）
                if scalar_data_mode == "sequence":
                    # 从序列中获取单个时间步
                    delta_ee = all_delta_actions[step]  # 当前时间步的数据
                    gripper = all_gripper_actions[step]
                    qpos = all_qpos[step]
                    ee_pose = all_ee_poses[step]
                    
                    # 确保数据是正确的单时间步形式
                    if isinstance(qpos, np.ndarray) and qpos.ndim > 1:
                        qpos = qpos.flatten()[:7]
                    
                    # 处理ee_pose - 转换为6D位姿
                    if isinstance(ee_pose, np.ndarray) and delta_ee.shape == (4, 4):
                        ee_pose = pose_to_6d(ee_pose)

                    # 处理delta_ee - 如果是4x4矩阵，也需要转换
                    if isinstance(delta_ee, np.ndarray) and delta_ee.shape == (4, 4):
                        delta_ee = pose_to_6d(delta_ee)
                        action = np.concatenate([delta_ee, [gripper]])
                    else:
                        action = np.concatenate([delta_ee[:6], [gripper]]) if len(delta_ee) >= 6 else np.zeros(7)

                    if isinstance(action, np.ndarray) and action.ndim > 1:
                        action = action.flatten()[:7]
                        
                    # 构建单一时间步的帧
                    frame = {
                        "observation.robot.qpos": qpos,  # [7]
                        "observation.robot.ee_pose": ee_pose,  # [6]
                        "action": action,  # [7]
                        "observation.images.Primary_0_0": primary_img,
                        "observation.images.Wrist_0_0": wrist_img
                    }
                    frames.append(frame)                
            except Exception as e:
                logging.warning(f"Error at step {step}: {str(e)}")
                continue

    # 返回处理后的数据
    if not frames:
        logging.warning(f"No valid frames in {episode_path.name}")
        return None
        
    return {
        "frames": frames,
        "videos": {
            "observation.images.Primary_0_0": episode_path/"observation/Primary_0_0/color_image/demo.mp4",
            "observation.images.Wrist_0_0": episode_path/"observation/Wrist_0_0/color_image/demo.mp4"
        },
        "meta": meta
    }



def load_local_dataset(episode_id: int, src_path: str) -> Optional[tuple]:
    try:
        # 处理7位数字目录格式（如果需要）
        episode_path = Path(src_path) / f"{episode_id:07d}"
        if not episode_path.exists():
            episode_path = Path(src_path) / f"{episode_id:06d}"  # 尝试6位格式
            
        if not episode_path.exists():
            logging.warning(f"Episode directory not found for ID {episode_id}")
            return None
            
        # 验证必要的路径
        if not (episode_path/"lmdb/data.mdb").exists():
            logging.warning(f"LMDB data not found for episode {episode_id}")
            return None
            
        # 加载元数据
        meta_path = episode_path/"meta_info.pkl"
        if not meta_path.exists():
            logging.warning(f"Meta info not found for episode {episode_id}")
            return None
            
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
            
        # 确保视频文件存在
        video_paths = [
            episode_path/"observation/Primary_0_0/color_image/demo.mp4",
            episode_path/"observation/Wrist_0_0/color_image/demo.mp4"
        ]
        
        if not all(p.exists() for p in video_paths):
            logging.warning(f"Video files missing for episode {episode_id}")
            return None
            
        # 加载LMDB数据
        return load_lmdb_data(episode_path, meta)
        
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
    dataset: BananaDataset,
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
                max_workers=os.cpu_count() // 2,
                desc=f"Loading chunk {chunk_start//chunk_size + 1}"
            )

        # 处理有效数据
        valid_episodes = [ep for ep in episodes_data if ep is not None]
        for episode_data in valid_episodes:
            # frames, videos, task_instruction = episode_data  # 修改此处
            # 正确的字典访问
            frames = episode_data["frames"]
            videos = episode_data["videos"]
            task_instruction = episode_data["meta"].get("language_instruction", "Manipulation Task")
            for frame in frames:
                dataset.add_frame(frame)
            # 使用从meta获取的指令    
            dataset.save_episode(task=task_instruction, videos=videos)
        
        # 内存管理
        del episodes_data, valid_episodes
        gc.collect()


# def process_episode(dataset, episode_data):
#     if episode_data is None:
#         return False
    
#     try:    
#         # 添加所有帧数据
#         for frame in episode_data["frames"]:
#             dataset.add_frame({
#                 "observation.robot.qpos": frame["observation.robot.qpos"],
#                 "observation.robot.ee_pose": frame["observation.robot.ee_pose"],
#                 "action": frame["action"],
#                 # 其他必要的字段
#             })
            
#         # 保存episode和视频
#         videos = {
#             "observation.images.Primary_0_0": episode_data["videos"]["observation.images.Primary_0_0"],
#             "observation.images.Wrist_0_0": episode_data["videos"]["observation.images.Wrist_0_0"]
#         }
        
#         # 获取任务指令（如果有）
#         task = episode_data["meta"].get("language_instruction", "Manipulation Task")
#         dataset.save_episode(task=task, videos=videos)
#         return True
        
#     except Exception as e:
#         logging.error(f"Error processing episode: {str(e)}")
#         return False

def process_episode(dataset, episode_data):
    if episode_data is None:
        return False
    
    try:    
        # 添加所有帧数据
        frames_added = 0
        for frame in episode_data["frames"]:
            try:
                dataset.add_frame({
                    "observation.robot.qpos": frame["observation.robot.qpos"],
                    "observation.robot.ee_pose": frame["observation.robot.ee_pose"],
                    "action": frame["action"],
                })
                frames_added += 1
            except Exception as frame_ex:
                logging.warning(f"添加帧时出错: {str(frame_ex)}")
                continue
        
        # 检查是否有帧被添加
        if frames_added == 0:
            logging.warning("没有成功添加任何帧，跳过当前episode")
            return False
            
        # 检查episode_buffer是否存在且size键存在
        if dataset.episode_buffer is None or "size" not in dataset.episode_buffer:
            logging.warning("episode_buffer不存在或缺少size字段")
            return False
            
        # 保存episode和视频
        videos = {
            "observation.images.Primary_0_0": episode_data["videos"]["observation.images.Primary_0_0"],
            "observation.images.Wrist_0_0": episode_data["videos"]["observation.images.Wrist_0_0"]
        }
        
        # 获取任务指令
        task = episode_data["meta"].get("language_instruction", "Manipulation Task")
        # logging.info(f"保存episode，包含{frames_added}个帧，任务：{task}")
        dataset.save_episode(task=task, videos=videos)
        return True
        
    except Exception as e:
        import traceback
        logging.error(f"处理episode时出错: {str(e)}\n{traceback.format_exc()}")
        return False

def main(
    src_path: str,
    tgt_path: str,
    repo_id: str = "banana_real/0010",
    debug: bool = False,
    chunk_size: int = 10
):
    # 初始化共享存储目录
    setup_directories()

    # 强制设置huggingface配置（保险措施）
    from datasets import config
    config.HF_DATASETS_CACHE = os.environ["HF_DATASETS_CACHE"]

    # 发现所有7位episode
    episode_dirs = sorted(Path(src_path).glob("[0-9]"*7))
    valid_episodes = [int(d.name) for d in episode_dirs if d.name.isdigit()]
    
    # 初始化数据集时修改robot_type
    dataset = BananaDataset.create(
        repo_id=repo_id,
        root=Path(tgt_path)/repo_id,
        fps=15,  # 修改为15 FPS
        robot_type="franka", 
        features=FEATURES,
    )
    for ep_id in tqdm(valid_episodes):
        try:
            data = load_local_dataset(ep_id, src_path)
            if data:
                success = process_episode(dataset, data)
                if not success:
                    logging.warning(f"处理episode {ep_id}失败")
            else:
                logging.warning(f"无法加载episode {ep_id}的数据")
        except Exception as e:
            logging.error(f"处理episode {ep_id}时发生异常: {str(e)}")    
    # for ep_id in tqdm(valid_episodes):
    #     data = load_local_dataset(ep_id, src_path)
    #     process_episode(dataset, data)
        
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
        default="banana_lerobot",
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