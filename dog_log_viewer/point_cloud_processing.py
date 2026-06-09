import logging

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot


def np_to_o3d(pc: np.ndarray) -> o3d.geometry.PointCloud:
    '''convert numpy (N, 3) ndarray to open3d'''
    assert pc.ndim == 2, f"{pc.shape=}"
    assert pc.shape[1] == 3, f"{pc.shape=}"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    return pcd


def o3d_to_np(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    '''convert open3d to numpy (N, 3) ndarray'''
    return np.asarray(pcd.points)


def transform_between_two_pcs(pc_src: np.ndarray, pc_dest: np.ndarray,
                              icp_nearest_neighbor: float = 0.2,
                              icp_num_its: int = 2000) -> np.ndarray:
    '''Returns T_src_dest, from which one can apply transform_pc(pc_dest, T_src_dest).'''
    pcd_src = np_to_o3d(pc_src)
    pcd_src.estimate_normals()
    pcd_dest = np_to_o3d(pc_dest)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_dest, pcd_src, icp_nearest_neighbor, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=icp_num_its)
    )

    return reg_p2p.transformation


def get_planar_equation(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
                        ) -> tuple[float, float, float, float] | None:
    '''From three points, get planar equation (Ax+By+Cz+d=0)'''

    assert p1.size == 3
    assert p2.size == 3
    assert p3.size == 3

    # Two direction vectors lying on the plane
    v1 = p2 - p1
    v2 = p3 - p1
    # Normal vector via cross product
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-6:
        return None
    # Plane equation: n · (x, y, z) + d = 0
    d = -n.dot(p1)
    return n[0], n[1], n[2], d


def voxel_filter(pts_np: np.ndarray, vox_size: float) -> np.ndarray:
    assert pts_np.ndim == 2
    assert pts_np.shape[1] == 3

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np)
    downpcd = pcd.voxel_down_sample(vox_size)

    return np.asarray(downpcd.points)


def detect_planar_patches(pts_np: np.ndarray):
    assert pts_np.ndim == 2
    assert pts_np.shape[1] == 3

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np)
    pcd.estimate_normals()

    oboxes = pcd.detect_planar_patches(
        normal_variance_threshold_deg=30,
        outlier_ratio=0.5,
        min_num_points=25
    )

    return [o for o in oboxes if np.min(o.extent) < 0.1]


def average_two_rotations(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    r1 = Rot.from_matrix(R1)
    r2 = Rot.from_matrix(R2)

    r_rel = r1.inv() * r2
    rotvec_rel = r_rel.as_rotvec()
    rotvec_half = Rot.from_rotvec(0.5 * rotvec_rel)

    r_avg = r1 * rotvec_half
    return r_avg.as_matrix()


def transform_pc(pc: np.ndarray, transform: np.ndarray) -> np.ndarray:
    '''Assume point cloud is (N, 3)'''
    # N, 3
    assert pc.shape[1] == 3
    R = transform[:3, :3]
    t = transform[:3, 3]

    out = (R @ pc.T).T
    out = out + t
    return out


def get_pixel_coords_and_depth(pc: np.ndarray, K: np.ndarray, HW: tuple[int, int]) -> np.ndarray:
    """
    Project camera frame points to image pixels.

    Parameters
    ----------
    pc : (N,3) array of (X,Y,Z) in camera coordinates
    K  : (3,3) intrinsic matrix
    HW

    Returns
    -------
    out : (M,3) array
          out[0] - pixel x, out[1] - pixel y, out[2] - depth
    """
    # Keep only points with positive depth
    z = pc[:, 2]
    valid = z > 0
    pc = pc[valid]
    z = z[valid]

    # Project to pixel coordinates
    xy = (K @ pc.T).T          # (N,3)
    u = xy[:, 0] / xy[:, 2]
    v = xy[:, 1] / xy[:, 2]

    # Keep points inside the image bounds
    h, w = HW
    in_image = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    u = u[in_image]
    v = v[in_image]
    z = z[in_image]

    out = np.column_stack([u, v, z])
    assert out.shape[1] == 3
    return out


def get_pc_as_depth_img(pc: np.ndarray, K: np.ndarray, HW: tuple[int, int]) -> np.ndarray:
    """
    Project a 3D point cloud (camera coordinates) into a depth image.

    Parameters
    ----------
    pc : (N,3) array of (X,Y,Z) in the camera coordinate system.
    K  : (3,3) intrinsic matrix.
    HW

    Returns
    -------
    depth_img : (height, width) array.
                depth_img[y,x] = distance along camera Z-axis of the
                nearest point that projects to pixel (x,y).
                Pixels without any point are set to np.inf.
    """
    h, w = HW
    # keep points in front of the camera
    Z = pc[:, 2]
    mask = Z > 0
    pc = pc[mask]
    Z = Z[mask]

    # project to pixel coordinates
    xy = (K @ pc.T).T          # (N,3)
    u = xy[:, 0] / xy[:, 2]
    v = xy[:, 1] / xy[:, 2]

    # keep points inside the image
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[inside].astype(int)
    v = v[inside].astype(int)
    Z = Z[inside]

    # initialise depth image with +inf
    depth = np.full(h * w, np.inf, dtype=np.float64)
    idx = v * w + u                # flatten pixel index

    # keep the smallest depth per pixel
    np.minimum.at(depth, idx, Z)
    depth[np.isinf(depth)] = np.nan

    # reshape
    depth_img = depth.reshape(h, w)
    return depth_img


def get_masked_points_cam_frame(depth_map: np.ndarray, mask: np.ndarray, K: np.ndarray,
                                HW: tuple[int, int]) -> np.ndarray:
    """
    Convert a depth image + binary mask into a 3D point cloud in the camera frame.

    Parameters
    ----------
    depth_map : (H, W) array
        Depth per pixel. Invalid depths are negative, ``np.inf`` or ``np.nan``.
    mask      : (H, W) boolean array
        True where the point is of interest.
    K         : (3, 3) intrinsic matrix
    HW        : (H, W) tuple of image dimensions (used for consistency checks)

    Returns
    -------
    out_pc : (N, 3) float64 array
        Each row is a 3D point [X, Y, Z] in camera coordinates.
    """
    H, W = HW
    assert depth_map.shape == (H, W)
    assert mask.shape == (H, W)

    # 1. Valid depth mask (positive, finite)
    depth_valid = (depth_map > 0) & np.isfinite(depth_map)

    # 2. Combined mask (user mask AND depth valid)
    valid = mask & depth_valid

    # 3. Pixel coordinates of valid points
    v, u = np.nonzero(valid)                 # v: row (y), u: column (x)
    Z = depth_map[v, u]                      # depth for each pixel

    # 4. Camera intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 5. Back‑project to 3‑D
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    out_pc = np.vstack((X, Y, Z)).T.astype(np.float64)
    assert out_pc.shape[1] == 3  # (N, 3)
    return out_pc


def get_dominant_cluster(pts_cam_frame: np.ndarray,
                         min_num_pts: int = 50) -> None | np.ndarray:
    logging.debug(f"Attempting to cluster pts of shape {pts_cam_frame.shape}")

    LIDAR_ANGULAR_RESOLUTION_DEG = 1.5
    '''Ouster has 64 beams over 90 deg FOV. 90deg/64 ~= 1.5'''
    EPSILON_MULT = 2.0
    '''For tolerance'''

    average_depth = np.median(np.linalg.norm(pts_cam_frame, axis=1))
    dbscan_eps = average_depth * np.sin(np.radians(LIDAR_ANGULAR_RESOLUTION_DEG)) * EPSILON_MULT
    logging.debug(f"With median depth of {average_depth:.2f}, dbscan_eps={dbscan_eps:.2f}")

    pcd = np_to_o3d(pts_cam_frame)
    labels = np.array(pcd.cluster_dbscan(
        eps=dbscan_eps, min_points=min_num_pts, print_progress=False))

    # No valid cluster
    if labels.size == 0:
        logging.debug("No input to DBSCAN, returning None")
        return None
    if labels.max() < 0:
        logging.debug("All points are noise, returning None")
        return None

    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_mask = unique_labels >= 0
    valid_labels = unique_labels[valid_mask]
    valid_counts = counts[valid_mask]

    # All points classified as noise
    if len(valid_labels) == 0:
        logging.debug("All points are noise, returning None")
        return None

    largest_cluster_label = valid_labels[np.argmax(valid_counts)]
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]

    largest_cluster_pcd = pcd.select_by_index(largest_cluster_indices)

    return o3d_to_np(largest_cluster_pcd)
