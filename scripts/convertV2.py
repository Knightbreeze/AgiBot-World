import argparse
import gc
import json
import shutil
import subprocess
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
import torchvision
from config import AgiBotWorld_CONFIG
from lerobot.common.datasets.compute_stats import auto_downsample_height_width, get_feature_stats, sample_indices
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import (
    check_timestamps_sync,
    get_episode_data_index,
    validate_episode_buffer,
    validate_frame,
)
from PIL import Image

torchvision.set_video_backend("pyav")


def generate_features_from_config():
    features = {}
    for key, value in AgiBotWorld_CONFIG["images"].items():
        features[f"observation.images.{key}"] = value
    for key, value in AgiBotWorld_CONFIG["states"].items():
        features[f"observation.states.{key}"] = value
    for key, value in AgiBotWorld_CONFIG["actions"].items():
        features[f"actions.{key}"] = value
    return features


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


def compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray], features: dict) -> dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue  # HACK: we should receive np.arrays of strings
        elif features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)
            axes_to_reduce = (0, 2, 3)  # keep channel dim
            keepdims = True
        else:
            ep_ft_array = data  # data is already a np.ndarray
            axes_to_reduce = 0  # compute stats over the first axis
            keepdims = data.ndim == 1  # keep as np.array

        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

        if features[key]["dtype"] in ["image", "video"]:
            # ep_stats[key] = {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()}
            value_norm = 1.0 if "depth" in key else 255.0
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / value_norm, axis=0) for k, v in ep_stats[key].items()
            }

    return ep_stats


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

    def save_episode(self, episode_data: dict | None = None, videos: dict | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key]).squeeze()

        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            episode_buffer[key] = str(video_path)  # PosixPath -> str
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(videos[key], video_path)

        ep_stats = compute_episode_stats(episode_buffer, self.features)

        self._save_episode_table(episode_buffer, episode_index)

        # `meta.save_episode` be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()

    def add_frame(self, frame: dict) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        features = {key: value for key, value in self.features.items() if key in self.hf_features}  # remove video keys
        validate_frame(frame, features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        # Add frame features to episode_buffer
        for key in frame:
            if key == "task":
                # Note: we associate the task in natural language to its task index during `save_episode`
                self.episode_buffer["task"].append(frame["task"])
                continue

            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            # if self.features[key]["dtype"] in ["image", "video"]:
            #     img_path = self._get_image_file_path(
            #         episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
            #     )
            #     if frame_index == 0:
            #         img_path.parent.mkdir(parents=True, exist_ok=True)
            #     self._save_image(frame[key], img_path)
            #     self.episode_buffer[key].append(str(img_path))
            # else:
            self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1


def load_depths(root_dir: str, camera_name: str):
    cam_path = Path(root_dir)
    all_imgs = sorted(list(cam_path.glob(f"{camera_name}*")))
    return [np.array(Image.open(f)).astype(np.float32)[:, :, None] / 1000 for f in all_imgs]


def load_local_dataset(episode_id: int, src_path: str, task_id: int, task_name: str, save_depth: bool) -> list | None:
    """Load local dataset and return a dict with observations and actions"""
    ob_dir = Path(src_path) / f"observations/{task_id}/{episode_id}"
    proprio_dir = Path(src_path) / f"proprio_stats/{task_id}/{episode_id}"

    state = {}
    action = {}
    with h5py.File(proprio_dir / "proprio_stats.h5", "r") as f:
        for key in AgiBotWorld_CONFIG["states"]:
            state[f"observation.states.{key}"] = np.array(f["state/" + key.replace(".", "/")], dtype=np.float32)
        for key in AgiBotWorld_CONFIG["actions"]:
            action[f"actions.{key}"] = np.array(f["action/" + key.replace(".", "/")], dtype=np.float32)

    num_frames = len(next(iter(action.values())))
    if save_depth:
        depth_imgs = load_depths(ob_dir / "depth", "head_depth")
        assert num_frames == len(depth_imgs), "Number of images and states are not equal"
    frames = [
        {
            **({"observation.images.head_depth": depth_imgs[i]} if save_depth else {}),
            **{key: value[i] for key, value in state.items()},
            **{key: value[i] for key, value in action.items()},
            "task": task_name,
        }
        for i in range(num_frames)
    ]

    v_path = ob_dir / "videos"
    videos = {
        f"observation.images.{key}": v_path / f"{key}_color.mp4"
        for key in AgiBotWorld_CONFIG["images"]
        if "depth" not in key
    }
    return frames, videos


def get_task_instruction(task_json_path: str) -> dict:
    """Get task language instruction"""
    with open(task_json_path, "r") as f:
        task_info = json.load(f)
    task_name = task_info[0]["task_name"]
    task_init_scene = task_info[0]["init_scene_text"]
    task_instruction = f"{task_name}.{task_init_scene}"
    print(f"Get Task Instruction <{task_instruction}>")
    return task_instruction


def get_all_tasks(src_path: Path, output_path: Path):
    json_files = src_path.glob("task_info/*.json")
    for json_file in json_files:
        local_dir = output_path / "agibotworld" / json_file.stem
        yield json_file, local_dir


def save_as_lerobot_dataset(task: tuple[Path, Path], num_threads, save_depth, debug):
    json_file, local_dir = task
    print(f"processing {json_file.stem}, saving to {local_dir}")
    src_path = json_file.parent.parent
    task_name = get_task_instruction(json_file)
    task_id = json_file.stem.split("_")[-1]
    features = generate_features_from_config()

    if local_dir.exists():
        shutil.rmtree(local_dir)

    if not save_depth:
        features.pop("observation.images.head_depth")

    dataset = AgiBotDataset.create(
        repo_id=json_file.stem,
        root=local_dir,
        fps=30,
        robot_type="a2d",
        features=features,
    )

    all_subdir = [f.as_posix() for f in src_path.glob(f"observations/{task_id}/*") if f.is_dir()]

    all_subdir_eids = [int(Path(path).name) for path in all_subdir]

    if debug or not save_depth:
        for eid in all_subdir_eids:
            raw_dataset = load_local_dataset(
                eid, src_path=src_path, task_id=task_id, task_name=task_name, save_depth=save_depth
            )
            frames, videos = raw_dataset
            for frame_data in frames:
                dataset.add_frame(frame_data)
            dataset.save_episode(videos=videos)
            gc.collect()
            # try:
            #     subprocess.run(command.format(local_dir, local_dir.name), shell=True, text=True, capture_output=True, check=True)
            # except subprocess.CalledProcessError as e:
            #     raise Exception(f"rclone failed: {str(e)}")
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for episode_id in all_subdir_eids:
                futures.append(
                    executor.submit(
                        load_local_dataset,
                        episode_id,
                        src_path=src_path,
                        task_id=task_id,
                        task_name=task_name,
                        save_depth=save_depth,
                    )
                )

            for raw_dataset in as_completed(futures):
                frames, videos = raw_dataset.result()
                for frame_data in frames:
                    dataset.add_frame(frame_data)
                dataset.save_episode(videos=videos)
                gc.collect()


def main(
    src_path: str,
    output_path: str,
    num_processes: int,
    num_threads: int,
    save_depth: bool,
    debug: bool = False,
):
    tasks = get_all_tasks(src_path, output_path)
    if debug:
        save_as_lerobot_dataset(next(tasks), num_threads=num_threads, save_depth=save_depth, debug=debug)
    else:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = {}
            for task in tasks:
                future = executor.submit(
                    save_as_lerobot_dataset, task, num_threads=num_threads, save_depth=save_depth, debug=debug
                )
                futures[future] = task[0].stem

            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    with open("output.txt", "a") as f:
                        f.write(f"exception happens as {task_id}, exception: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(**vars(args))
