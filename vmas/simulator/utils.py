#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple, Union, Dict, Sequence

import numpy as np
import torch
from torch import Tensor

X = 0
Y = 1
Z = 2
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
VIEWER_MIN_ZOOM = 1.2
INITIAL_VIEWER_SIZE = (700, 700)
LINE_MIN_DIST = 4 / 6e2
COLLISION_FORCE = 100
JOINT_FORCE = 130

DRAG = 0.25
LINEAR_FRICTION = 0.0
ANGULAR_FRICTION = 0.0

DEVICE_TYPING = Union[torch.device, str, int]

VIRIDIS_CMAP = np.array(
    [
        [0.267004, 0.004874, 0.329415],
        [0.278826, 0.17549, 0.483397],
        [0.229739, 0.322361, 0.545706],
        [0.172719, 0.448791, 0.557885],
        [0.127568, 0.566949, 0.550556],
        [0.157851, 0.683765, 0.501686],
        [0.369214, 0.788888, 0.382914],
        [0.678489, 0.863742, 0.189503],
    ]
)

AGENT_OBS_TYPE = Union[Tensor, Dict[str, Tensor]]
AGENT_INFO_TYPE = Dict[str, Tensor]
AGENT_REWARD_TYPE = Tensor

OBS_TYPE = Union[List[AGENT_OBS_TYPE], Dict[str, AGENT_OBS_TYPE]]
INFO_TYPE = Union[List[AGENT_INFO_TYPE], Dict[str, AGENT_INFO_TYPE]]
REWARD_TYPE = Union[List[AGENT_REWARD_TYPE], Dict[str, AGENT_REWARD_TYPE]]
DONE_TYPE = Tensor


class Color(Enum):
    RED = (0.75, 0.25, 0.25)
    GREEN = (0.25, 0.75, 0.25)
    BLUE = (0.25, 0.25, 0.75)
    LIGHT_GREEN = (0.45, 0.95, 0.45)
    WHITE = (0.75, 0.75, 0.75)
    GRAY = (0.25, 0.25, 0.25)
    BLACK = (0.15, 0.15, 0.15)
    ORANGE = (0.95, 0.65, 0.25)
    YELLOW = (0.95, 0.95, 0.25)
    PURPLE = (0.75, 0.25, 0.75)
    PINK = (0.95, 0.25, 0.75)
    BROWN = (0.65, 0.45, 0.25)
    LIGHT_BLUE = (0.25, 0.65, 0.95)
    LIGHT_GRAY = (0.65, 0.65, 0.65)
    LIGHT_RED = (0.95, 0.45, 0.45)
    LIGHT_PURPLE = (0.65, 0.25, 0.65)
    LIGHT_PINK = (0.95, 0.25, 0.65)
    LIGHT_BROWN = (0.75, 0.45, 0.25)
    LIGHT_ORANGE = (0.95, 0.45, 0.25)
    LIGHT_YELLOW = (0.95, 0.95, 0.45)


def override(cls):
    """Decorator for documenting method overrides."""

    def check_override(method):
        if method.__name__ not in dir(cls):
            raise NameError("{} does not override any method of {}".format(method, cls))
        return method

    return check_override


def _init_pyglet_device():
    available_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if available_devices is not None and len(available_devices) > 0:
        os.environ["PYGLET_HEADLESS_DEVICE"] = (
            available_devices.split(",")[0]
            if len(available_devices) > 1
            else available_devices
        )


class Observable(ABC):
    def __init__(self):
        self._observers = []

    def subscribe(self, observer):
        self._observers.append(observer)

    def notify_observers(self, *args, **kwargs):
        for obs in self._observers:
            obs.notify(self, *args, **kwargs)

    def unsubscribe(self, observer):
        self._observers.remove(observer)


class Observer(ABC):
    @abstractmethod
    def notify(self, observable, *args, **kwargs):
        raise NotImplementedError


def save_video(name: str, frame_list: List[np.array], fps: int):
    """Requres cv2"""
    import cv2

    video_name = name + ".mp4"

    # Produce a video
    video = cv2.VideoWriter(
        video_name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,  # FPS
        (frame_list[0].shape[1], frame_list[0].shape[0]),
    )
    for img in frame_list:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f"{name}.png", img)
        # break
        video.write(img)
    video.release()


def x_to_rgb_colormap(
        x: np.ndarray, low: float = None, high: float = None, alpha: float = 1.0
):
    res = VIRIDIS_CMAP.shape[0]
    if low is None:
        low = np.min(x)
    if high is None:
        high = np.max(x)
    x = np.clip(x, low, high)
    x = (x - low) / (high - low) * (res - 1)
    x_c0_idx = np.floor(x).astype(int)
    x_c1_idx = np.ceil(x).astype(int)
    x_c0 = VIRIDIS_CMAP[x_c0_idx, :]
    x_c1 = VIRIDIS_CMAP[x_c1_idx, :]
    t = x - x_c0_idx
    rgb = t[:, None] * x_c1 + (1 - t)[:, None] * x_c0
    colors = np.concatenate([rgb, alpha * np.ones((rgb.shape[0], 1))], axis=-1)
    return colors


def extract_nested_with_index(data: Union[Tensor, Dict[str, Tensor]], index: int):
    if isinstance(data, Tensor):
        return data[index]
    elif isinstance(data, Dict):
        return {
            key: extract_nested_with_index(value, index) for key, value in data.items()
        }
    else:
        raise NotImplementedError(f"Invalid type of data {data}")


class TorchUtils:
    @staticmethod
    def clamp_with_norm(tensor: Tensor, max_norm: float):
        norm = torch.linalg.vector_norm(tensor, dim=-1)
        new_tensor = (tensor / norm.unsqueeze(-1)) * max_norm
        tensor[norm > max_norm] = new_tensor[norm > max_norm]
        return tensor

    @staticmethod
    def rotate_vector(vector: Tensor, angle: Tensor):
        if len(angle.shape) > 1:
            angle = angle.squeeze(-1)
        if len(vector.shape) == 1:
            vector = vector.unsqueeze(0)
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        return torch.stack(
            [
                vector[:, X] * cos - vector[:, Y] * sin,
                vector[:, X] * sin + vector[:, Y] * cos,
            ],
            dim=-1,
        )

    @staticmethod
    def cross(vector_a: Tensor, vector_b: Tensor):
        return (
                vector_a[:, X] * vector_b[:, Y] - vector_a[:, Y] * vector_b[:, X]
        ).unsqueeze(1)

    @staticmethod
    def compute_torque(f: Tensor, r: Tensor) -> Tensor:
        return TorchUtils.cross(r, f)

    @staticmethod
    def to_numpy(data: Union[Tensor, Dict[str, Tensor], List[Tensor]]):
        if isinstance(data, Tensor):
            return data.cpu().detach().numpy()
        elif isinstance(data, Dict):
            return {key: TorchUtils.to_numpy(value) for key, value in data.items()}
        elif isinstance(data, Sequence):
            return [TorchUtils.to_numpy(value) for value in data]
        else:
            raise NotImplementedError(f"Invalid type of data {data}")


class ScenarioUtils:
    @staticmethod
    def spawn_entities_randomly(
            entities,
            world,
            env_index: int,
            min_dist_between_entities: float,
            x_bounds: Tuple[int, int],
            y_bounds: Tuple[int, int],
            occupied_positions: Tensor = None,
    ):
        batch_size = world.batch_dim if env_index is None else 1

        if occupied_positions is None:
            occupied_positions = torch.zeros(
                (batch_size, 0, world.dim_p), device=world.device
            )

        for entity in entities:
            pos = ScenarioUtils.find_random_pos_for_entity(
                occupied_positions,
                env_index,
                world,
                min_dist_between_entities,
                x_bounds,
                y_bounds,
            )
            occupied_positions = torch.cat([occupied_positions, pos], dim=1)
            entity.set_pos(pos.squeeze(1), batch_index=env_index)

    @staticmethod
    def find_random_pos_for_entity(
            occupied_positions: Union[torch.Tensor, None],
            env_index: int,
            world,
            min_dist_between_entities: float,
            x_bounds: Tuple[int, int],
            y_bounds: Tuple[int, int],
    ):
        batch_size = world.batch_dim if env_index is None else 1

        pos = None
        while True:
            proposed_pos = torch.cat(
                [
                    torch.empty(
                        (batch_size, 1, 1),
                        device=world.device,
                        dtype=torch.float32,
                    ).uniform_(*x_bounds),
                    torch.empty(
                        (batch_size, 1, 1),
                        device=world.device,
                        dtype=torch.float32,
                    ).uniform_(*y_bounds),
                ],
                dim=2,
            )
            if pos is None:
                pos = proposed_pos
            if occupied_positions is None or occupied_positions.shape[1] == 0:
                break

            dist = torch.cdist(occupied_positions, pos)
            overlaps = torch.any((dist < min_dist_between_entities).squeeze(2), dim=1)
            if torch.any(overlaps, dim=0):
                pos[overlaps] = proposed_pos[overlaps]
            else:
                break
        return pos

    @staticmethod
    def format_obs(obs, index: int = None):
        if index is not None:
            if isinstance(obs, Tensor):
                return list(np.around(obs[index].cpu().tolist(), decimals=2))
            elif isinstance(obs, Dict):
                return {key: ScenarioUtils.format_obs(value, index) for key, value in obs.items()}
            else:
                raise NotImplementedError(f"Invalid type of observation {obs}")
        else:
            if isinstance(obs, Tensor):
                return list(np.around(obs.cpu().tolist(), decimals=2))
            elif isinstance(obs, Dict):
                return {key: ScenarioUtils.format_obs(value) for key, value in obs.items()}
            else:
                raise NotImplementedError(f"Invalid type of observation {obs}")

    @staticmethod
    def create_distance_field_torch(occupied_positions, x_bounds, y_bounds, resolution=100):
        """
        Create a distance field using PyTorch.
        :param occupied_positions: Tensor of positions [N, 2].
        :param x_bounds: Tuple (min_x, max_x).
        :param y_bounds: Tuple (min_y, max_y).
        :param resolution: Resolution of the grid.
        """
        # Create a grid of points
        device = occupied_positions.device
        x = torch.linspace(x_bounds[0], x_bounds[1], resolution, device=device)
        y = torch.linspace(y_bounds[0], y_bounds[1], resolution, device=device)
        xx, yy = torch.meshgrid(x, y)
        grid_points = torch.stack((xx.flatten(), yy.flatten()), dim=1)

        # Calculate distance to the nearest occupied position
        distances = torch.cdist(grid_points, occupied_positions).min(dim=1).values
        distance_grid = distances.view(resolution, resolution)

        return distance_grid

    @staticmethod
    def find_position_torch(distance_field, x_bounds, y_bounds, steps=100, learning_rate=1e-1):
        """
        Find a position in the distance field with maximum distance from occupied positions.
        :param distance_field: The distance field tensor.
        :param x_bounds: Tuple (min_x, max_x).
        :param y_bounds: Tuple (min_y, max_y).
        :param steps: Number of optimization steps.
        :param learning_rate: Learning rate for the optimization.
        """
        device = distance_field.device
        # Random initial guess
        pos = torch.tensor([np.random.uniform(*x_bounds), np.random.uniform(*y_bounds)], requires_grad=True,
                           device=device)

        optimizer = torch.optim.Adam([pos], lr=learning_rate)

        for _ in range(steps):
            optimizer.zero_grad()

            # Clamp position within bounds
            clamped_pos = torch.stack([pos[0].clamp(*x_bounds), pos[1].clamp(*y_bounds)])

            # Interpolate distance at the current position
            distance = torch.nn.functional.grid_sample(
                distance_field.unsqueeze(0).unsqueeze(0),
                clamped_pos.view(1, 1, 1, 2) * 2 - 1,  # Normalize to [-1, 1]
                padding_mode='border',
                align_corners=True
            ).squeeze()

            # Maximize the distance (minimize the negative distance)
            loss = -distance
            loss.backward()
            optimizer.step()

        return pos.detach(), distance.detach()

    @staticmethod
    def find_position_torch_adjusted(distance_field, occupied_positions, x_bounds, y_bounds, steps=100,
                                     learning_rate=1e-1):
        device = distance_field.device
        pos = torch.tensor([np.random.uniform(*x_bounds), np.random.uniform(*y_bounds)], requires_grad=True,
                           device=device)
        optimizer = torch.optim.Adam([pos], lr=learning_rate)

        for _ in range(steps):
            optimizer.zero_grad()
            clamped_pos = torch.stack([pos[0].clamp(*x_bounds), pos[1].clamp(*y_bounds)])
            normalized_pos = (clamped_pos - torch.tensor([x_bounds[0], y_bounds[0]], device=device)) / torch.tensor(
                [x_bounds[1] - x_bounds[0], y_bounds[1] - y_bounds[0]], device=device)
            distance = torch.nn.functional.grid_sample(distance_field.unsqueeze(0).unsqueeze(0),
                                                       normalized_pos.view(1, 1, 1, 2) * 2 - 1, padding_mode='border',
                                                       align_corners=True).squeeze()
            loss = -distance
            loss.backward()
            optimizer.step()

        final_pos = clamped_pos.detach()
        actual_distance = torch.cdist(occupied_positions, final_pos.unsqueeze(0)).min()
        return final_pos, actual_distance.item()

    @staticmethod
    def try_find_random_pos_for_entity(
            occupied_positions: Union[torch.Tensor, None],
            env_index: int,
            world,
            min_dist_between_entities: float,
            x_bounds: Tuple[int, int],
            y_bounds: Tuple[int, int],
            num_tries: int = 100,
    ):
        batch_size = world.batch_dim if env_index is None else 1

        # Example Usage
        distance_field = ScenarioUtils.create_distance_field_torch(occupied_positions, x_bounds, y_bounds,
                                                                   resolution=1000)

        pos = None
        epoch = 0
        while True:
            proposed_pos, dist = ScenarioUtils.find_position_torch_adjusted(distance_field, occupied_positions,
                                                                            x_bounds, y_bounds)
            proposed_pos = proposed_pos.view(1, 1, 2)
            print(proposed_pos)

            proposed_pos2 = torch.cat(
                [
                    torch.empty(
                        (batch_size, 1, 1),
                        device=world.device,
                        dtype=torch.float32,
                    ).uniform_(*x_bounds),
                    torch.empty(
                        (batch_size, 1, 1),
                        device=world.device,
                        dtype=torch.float32,
                    ).uniform_(*y_bounds),
                ],
                dim=2,
            )
            if pos is None:
                pos = proposed_pos
            if occupied_positions is None or occupied_positions.shape[1] == 0:
                break

            dist = torch.cdist(occupied_positions, pos)
            overlaps = torch.any((dist < min_dist_between_entities).squeeze(2), dim=1)
            if torch.any(overlaps, dim=0):
                pos[overlaps] = proposed_pos[overlaps]
            else:
                break
            epoch += 1
            if epoch > num_tries:
                print("Failed to find a free position")
                return None
        return pos
