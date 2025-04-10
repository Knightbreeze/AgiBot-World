import argparse
import json
import logging
import os
import shutil
from concurrent.futures import ALL_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torchvision
import cv2
import h5py
import lmdb
import numpy as np
import pickle
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from lerobot.common.datasets.compute_stats import auto_downsample_height_width, get_feature_stats, sample_indices
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import (
    check_timestamps_sync,
    get_episode_data_index,
    validate_episode_buffer,
    validate_frame,
)

torchvision.set_video_backend("pyav")

# 视频文件名常量
PRIMARY_COLOR = "Primary_0_0_color.mp4"
WRIST_COLOR = "Wrist_0_0_color.mp4"

FEATURES = {
    "observation.images.Primary_0_0": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.Wrist_0_0": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.robot.qpos": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["arm_0", "arm_1", "arm_2", "arm_3", "arm_4", "arm_5", "arm_6"],
    },
    "observation.robot.ee_pose": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "z", "roll", "pitch", "yaw"],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    },
}


def pose_to_6d(pose, degrees=False):
    """将4x4矩阵转换为6D位姿表示(xyz+欧拉角)"""
    pose6d = np.zeros(6)
    pose6d[:3] = pose[:3, 3]  # 提取位置（平移向量）
    pose6d[3:6] = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=degrees)  # 提取旋转（欧拉角）
    return pose6d


def sample_images(input):
    if type(input) is str:
        video_path = input
        reader = torchvision.io.VideoReader(video_path, stream="video")
        frames = [frame["data"] for frame in reader]
        frames_array = torch.stack(frames).numpy()  # Shape: [T, C, H, W]

        sampled_indices = sample_indices(len(frames_array))
        images = None
        for i, idx in enumerate(sampled_indices):
            img = frames_array[idx]
            img = auto_downsample_height_width(img)

            if images is None:
                images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

            images[i] = img
    elif type(input) is np.ndarray:
        frames_array = input[:, None, :, :]  # Shape: [T, C, H, W]
        sampled_indices = sample_indices(len(frames_array))
        images = None
        for i, idx in enumerate(sampled_indices):
            img = frames_array[idx]
            img = auto_downsample_height_width(img)

            if images is None:
                images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

            images[i] = img

    return images


def compute_episode_stats(episode_data: Dict[str, List[str] | np.ndarray], features: Dict) -> Dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue
        elif features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)
            axes_to_reduce = (0, 2, 3)  # keep channel dim
            keepdims = True
        else:
            ep_ft_array = data  # data is already a np.ndarray
            axes_to_reduce = 0  # compute stats over the first axis
            keepdims = data.ndim == 1  # keep as np.array

        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

        # 标准化视频数据
        if features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
            }

    return ep_stats


class BananaDataset(LeRobotDataset):
    """处理Banana数据集的自定义LeRobotDataset子类"""
    
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

    def save_episode(self, episode_data: dict | None = None, videos: dict | None = None) -> None:
        """保存当前episode到磁盘，复制视频文件到目标位置"""
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # 处理特殊字段
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # 将任务添加到任务字典
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # 将任务文本转换为索引
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        # 处理非视频数据
        for key, ft in self.features.items():
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key]).squeeze()

        # 复制视频文件
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            episode_buffer[key] = str(video_path)  # PosixPath -> str
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(videos[key], video_path)

        # 计算统计数据
        ep_stats = compute_episode_stats(episode_buffer, self.features)

        # 保存episode数据
        self._save_episode_table(episode_buffer, episode_index)
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        # 验证时间戳同步
        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        # 重置buffer
        if not episode_data:
            self.episode_buffer = self.create_episode_buffer()

    def add_frame(self, frame: dict) -> None:
        """添加帧数据到episode buffer"""
        # 将Torch张量转换为numpy（如果需要）
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        # 仅保留非视频特征用于验证
        features = {key: value for key, value in self.features.items() if key in self.hf_features}
        # validate_frame(frame, features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # 自动添加帧索引和时间戳
        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        # 添加帧特征到episode buffer
        for key in frame:
            if key == "task":
                self.episode_buffer["task"].append(frame["task"])
                continue

            if key not in self.features:
                raise ValueError(f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'.")
            
            self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1


def load_lmdb_data(episode_path: Path) -> Optional[Dict]:
    """从LMDB加载一个episode的数据"""
    try:
        env = lmdb.open(
            str(episode_path / "lmdb"),
            readonly=True,
            lock=False,
            max_readers=128,
            readahead=False
        )
        
        frames = []
        with env.begin(write=False) as txn:
            # 默认任务指令
            task_name = "Grasp the brush and empty the objects on the chopping board into the dustpan."
            
            # 获取所有键
            cursor = txn.cursor()
            keys = [k for k, _ in cursor]
            
            # 确定主相机图像数量
            primary_keys = sorted([k for k in keys if b'Primary_0_0/color_image' in k])
            if not primary_keys:
                return None
                
            total_steps = len(primary_keys)
            
            # 检查是否以序列形式存储数据
            scalar_keys = [b'delta_arm_ee_action', b'gripper_action', 
                         b'observation/robot/qpos', b'observation/robot/forlan2robot_pose']
                         
            scalar_data_mode = "sequence" if all(k in keys for k in scalar_keys) else "step"
            
            # 如果数据以序列形式存储，预先加载
            if scalar_data_mode == "sequence":
                all_delta_actions = pickle.loads(txn.get(b'delta_arm_ee_action'))
                all_gripper_actions = pickle.loads(txn.get(b'gripper_action'))
                all_qpos = pickle.loads(txn.get(b'observation/robot/qpos'))
                all_ee_poses = pickle.loads(txn.get(b'observation/robot/forlan2robot_pose'))
               # 确保所有序列长度足够
                min_len = min(len(all_delta_actions), len(all_gripper_actions), len(all_qpos), len(all_ee_poses))
                if min_len < total_steps:
                    print(f"调整步数: {total_steps} -> {min_len}")
                    total_steps = min_len
            # 处理每一步
            for step in range(total_steps):
                step_str = f"{step:04d}"
                
                # 构建图像键
                primary_img_key = f"observation/Primary_0_0/color_image/{step_str}".encode()
                wrist_img_key = f"observation/Wrist_0_0/color_image/{step_str}".encode()
                
                # 确保图像键存在
                if primary_img_key not in keys or wrist_img_key not in keys:
                    continue
                
                # 加载图像
                primary_img = cv2.imdecode(pickle.loads(txn.get(primary_img_key)), cv2.IMREAD_COLOR)
                wrist_img = cv2.imdecode(pickle.loads(txn.get(wrist_img_key)), cv2.IMREAD_COLOR)
                
                # 加载标量数据
                if scalar_data_mode == "sequence":
                    delta_ee = all_delta_actions[step]
                    gripper = all_gripper_actions[step]
                    qpos = all_qpos[step]
                    ee_pose = all_ee_poses[step]
                    
                    # 处理ee_pose - 转换为6D位姿
                    if isinstance(ee_pose, np.ndarray) and ee_pose.shape == (4, 4):
                        ee_pose = pose_to_6d(ee_pose)
                        
                    # 创建动作数组
                    if isinstance(delta_ee, np.ndarray):
                        if delta_ee.shape == (4, 4):
                            delta_ee = pose_to_6d(delta_ee)
                            action = np.concatenate([delta_ee, [gripper]])
                        else:
                            action = np.concatenate([delta_ee[:6], [gripper]])
                    else:
                        action = np.zeros(7)
                        
                    # 创建帧数据
                    frame = {
                        "observation.robot.qpos": qpos[:7],
                        "observation.robot.ee_pose": ee_pose[:6],
                        "action": action[:7],
                        "observation.images.Primary_0_0": primary_img,
                        "observation.images.Wrist_0_0": wrist_img,
                        "task": task_name
                    }
                    frames.append(frame)
                    
        # 如果没有成功加载任何帧，返回None
        if not frames:
            return None
            
        # 返回处理后的数据
        return {
            "frames": frames,
            "videos": {
                "observation.images.Primary_0_0": episode_path / "observation/Primary_0_0/color_image/demo.mp4",
                "observation.images.Wrist_0_0": episode_path / "observation/Wrist_0_0/color_image/demo.mp4"
            },
        }
        
    except Exception as e:
        print(f"Error loading episode data: {str(e)}")
        return None


def load_local_dataset(episode_id: int, src_path: str) -> Optional[Dict]:
    """加载本地数据集并返回包含观测和动作的字典"""
    # 处理不同格式的目录名
    for format_str in [f"{episode_id:07d}", f"{episode_id:06d}", str(episode_id)]:
        episode_path = Path(src_path) / format_str
        if episode_path.exists():
            break
    else:
        logging.warning(f"Episode directory not found for ID {episode_id}")
        return None
        
    if not episode_path.exists():
        return None
        
    # 验证必要的路径
    if not (episode_path / "lmdb/data.mdb").exists():
        logging.warning(f"LMDB data not found for episode {episode_id}")
        return None
        
    # 检查视频文件是否存在
    video_paths = [
        episode_path / "observation/Primary_0_0/color_image/demo.mp4",
        episode_path / "observation/Wrist_0_0/color_image/demo.mp4"
    ]
    
    if not all(p.exists() for p in video_paths):
        logging.warning(f"Video files missing for episode {episode_id}")
        return None
        
    # 加载LMDB数据
    raw_dataset = load_lmdb_data(episode_path)
    frames = raw_dataset["frames"]
    videos = raw_dataset["videos"]
    return frames,videos


def get_task_instruction(task_json_path: str) -> dict:
    """Get task language instruction"""
    with open(task_json_path, "r") as f:
        task_info = json.load(f)
    task_name = task_info[0]["task_name"]
    task_init_scene = task_info[0]["init_scene_text"]
    task_instruction = f"{task_name}.{task_init_scene}"
    print(f"Get Task Instruction <{task_instruction}>")
    return task_instruction


def save_as_lerobot_dataset(task: tuple[Path, Path], num_threads, debug):
    src_path, output_dir = task
    episode_id = src_path.name
    print(f"Processing episode {episode_id}, saving to {output_dir}")
    
    if output_dir.exists():
        shutil.rmtree(output_dir)

    dataset = BananaDataset.create(
        repo_id=f"banana_episode_{episode_id}",
        root=output_dir,
        fps=15,
        robot_type="franka",
        features=FEATURES,
    )

    all_subdir = sorted([f.as_posix() for f in src_path.glob(f"*") if f.is_dir()])

    all_subdir_eids = [int(Path(path).name) for path in all_subdir]

    if debug:
        frames,videos = load_local_dataset(all_subdir_eids[0], src_path=src_path)
        for frame_data in frames:
            dataset.add_frame(frame_data)
        dataset.save_episode(videos=videos)
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for episode_id in all_subdir_eids:
                futures.append(
                    executor.submit(load_local_dataset, episode_id, src_path=src_path)
                )

            for raw_dataset in as_completed(futures):
                frames,videos = raw_dataset.result()
                for frame_data in frames:
                    dataset.add_frame(frame_data)
                dataset.save_episode(videos=videos)


def get_all_tasks(src_path: Path, output_path: Path):
    # json_files = src_path.glob("task_info/*.json")
    # for json_file in json_files:
    #     local_dir = output_path / "banana" / json_file.stem
    #     yield json_file, local_dir
    src_dir_name = src_path.name
    local_dir = output_path / src_dir_name
    yield src_path, local_dir


def main(
    src_path: str,
    output_path: str,
    num_processes: int,
    num_threads: int,
    debug: bool = False,
):
    logging.info("Scanning for episodes...")
    tasks = get_all_tasks(src_path, output_path)
    # logging.info(f"Found {len(tasks)} episodes")
    if debug:
        task = next(tasks)
        save_as_lerobot_dataset(task, num_threads=num_threads, debug=debug)
    else:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(save_as_lerobot_dataset, task, num_threads=num_threads, debug=debug) for task in tasks
            ]
            wait(futures, return_when=ALL_COMPLETED)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-path",
        type=Path,
        required=True,
        help="Path to source data directory",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to output directory",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with limited episodes",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Number of processes to use",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="Number of threads per process",
    )
    args = parser.parse_args()

    main(**vars(args))