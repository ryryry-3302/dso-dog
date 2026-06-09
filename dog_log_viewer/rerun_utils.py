from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import rerun as rr
from scipy.spatial.transform import Rotation as R


def _create_origin_axes(length: float = 1.0) -> rr.LineStrips3D:
    return rr.LineStrips3D(
        strips=[np.array([[0, 0, 0], [length, 0, 0]]),
                np.array([[0, 0, 0], [0, length, 0]]),
                np.array([[0, 0, 0], [0, 0, length]])],
        colors=[np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0])])


def _T_to_rerun(T: np.ndarray) -> rr.Transform3D:
    assert T.shape == (4, 4)
    rr_trans = T[:3, 3]
    rw, rx, ry, rz = pr.quaternion_from_matrix(T[:3, :3])
    rr_quat = rr.Quaternion(xyzw=(rx, ry, rz, rw))

    return rr.Transform3D(translation=rr_trans, quaternion=rr_quat)


def _create_path(transforms: list[np.ndarray],
                 color: Optional[tuple[float,
                                       float,
                                       float]] = None) -> rr.LineStrips3D:
    path = [T[:3, 3] for T in transforms]

    if color is None:
        color = (0.0, 1.0, 0.0)
    return rr.LineStrips3D(strips=[np.array(path)], colors=[np.array(color)])


def _K_arr_to_mtx(k_flat: tuple[float, float, float, float]
                  | list[float] | np.ndarray) -> np.ndarray:
    assert len(k_flat) == 4
    K = np.eye(3)
    K[0, 0] = k_flat[0]
    K[1, 1] = k_flat[1]
    K[0, 2] = k_flat[2]
    K[1, 2] = k_flat[3]
    return K


def get_line_strips(cumulative_path):
    line_strips = []
    if len(cumulative_path) == 0:
        return []
    if len(cumulative_path) == 1:
        return [[cumulative_path[0], cumulative_path[0]]]

    for i in range(1, len(cumulative_path)):
        line_strips.append([
            cumulative_path[i - 1], cumulative_path[i]
        ])

    return line_strips


def get_color_vals(traj: list, cmap: str, min_alpha: float):
    """
    Return RGBA colors for *traj*.
    - Colors come from *cmap*.
    - Alpha rises linearly from *min_alpha* to 1 over the list.
    """
    n = len(traj)
    if n == 0:
        return []

    if not (0.0 <= min_alpha <= 1.0):
        raise ValueError("min_alpha must be in [0, 1]")

    # Normalized indices in [0, 1]
    indices = [i / (n - 1) if n > 1 else 0.0 for i in range(n)]

    cmap_obj = plt.get_cmap(cmap)

    colors: list[tuple[float, float, float, float]] = []
    for idx in indices:
        r, g, b, _ = cmap_obj(idx)  # get RGBA; ignore original alpha
        a = min_alpha + (1.0 - min_alpha) * idx
        colors.append((r, g, b, a))

    return colors


def color_pc(pc: np.ndarray,
             alpha: float = 1.0,
             cmap="turbo",
             func: Callable[[np.ndarray], np.ndarray] = lambda x: x[:, 2]):
    """
    Adds a per-point coloring to the point cloud.

    By default, colors by the z-hieght of the points.
    """
    vals = func(pc)
    cmap = plt.get_cmap(cmap)
    vmin, vmax = np.min(vals), np.max(vals)
    vals_norm = (vals - vmin) / (vmax - vmin)
    colors = cmap(vals_norm)
    colors[:, -1] = alpha
    return colors


def log_position(T_world_body: np.ndarray,
                 cumulative_path: list[np.ndarray],
                 pose_label="pose",
                 world_frame_label="world",
                 body_frame_label="imu") -> list[np.ndarray]:
    rr.log(f"/{pose_label}/position/x", rr.Scalars(T_world_body[0, 3]))
    rr.log(f"/{pose_label}/position/y", rr.Scalars(T_world_body[1, 3]))
    rr.log(f"/{pose_label}/position/z", rr.Scalars(T_world_body[2, 3]))
    yaw, pitch, roll = R.from_matrix(
        T_world_body[:3, :3]).as_euler('zyx', degrees=True)
    rr.log(f"/{pose_label}/rotation/yaw", rr.Scalars(yaw))
    rr.log(f"/{pose_label}/rotation/pitch", rr.Scalars(pitch))
    rr.log(f"/{pose_label}/rotation/roll", rr.Scalars(roll))
    cumulative_path.append(T_world_body[:3, 3])
    linestrip = get_line_strips(cumulative_path)
    rgba = get_color_vals(linestrip, "viridis", 0.1)
    assert len(linestrip) == len(rgba)

    rr.log(f"/{world_frame_label}/path",
           rr.LineStrips3D(strips=linestrip, colors=rgba))
    rr.log(f"/{world_frame_label}/{body_frame_label}",
           rr.Transform3D(translation=T_world_body[:3, 3],
                          mat3x3=T_world_body[:3, :3]))

    return cumulative_path


def _K_arr_to_mtx(k_flat: tuple[float, float, float, float]
                  | list[float] | np.ndarray) -> np.ndarray:
    assert len(k_flat) == 4
    K = np.eye(3)
    K[0, 0] = k_flat[0]
    K[1, 1] = k_flat[1]
    K[0, 2] = k_flat[2]
    K[1, 2] = k_flat[3]
    return K
