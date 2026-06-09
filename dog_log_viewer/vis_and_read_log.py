import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import rerun as rr
from tqdm import tqdm

import point_cloud_processing as PCP
import rerun_utils as RRU

sys.path.append("./pygenerictools")
from pygenericdata import utils  # noqa
from pygenericdata.camera import reader  # noqa
from read_xml_calib import readXmlCameraCal, readXmlExtrinsicCal  # noqa
from readLidarBinaryFormat import ScanFile  # noqa

# Rerun frame names viewport
WORLD_FRAME = "/world"
BODY_FRAME = f"{WORLD_FRAME}/body"
CAMERA_FRAME = f"{BODY_FRAME}/camera"
IMAGE_FRAME = f"{CAMERA_FRAME}/image"
DETECTION_BOX_FRAME = f"{CAMERA_FRAME}/detections"
DEPTH_FRAME = f"{CAMERA_FRAME}/depth"
SEGMENTATION_FRAME = f"{CAMERA_FRAME}/segmentation"
LIDAR_FRAME = f"{BODY_FRAME}/lidar"
DETECTION_PC_FRAME = f"{WORLD_FRAME}/detection_pc"
TIMELINE = "system_time"


def main(camera_cal_path: str | Path, lidar_cal_path: str | Path,
         gclf_file_path: str | Path, lidar_log_path: str | Path,
         lidar_id: int = 71, camera_id: int = 211):
    # Input validation
    camera_cal_path = Path(camera_cal_path)
    assert camera_cal_path.exists(), f"camera_cal_path [{str(camera_cal_path)}] does not exist."
    lidar_cal_path = Path(lidar_cal_path)
    assert lidar_cal_path.exists(), f"lidar_cal_path [{str(lidar_cal_path)}] does not exist."
    gclf_file_path = Path(gclf_file_path)
    assert gclf_file_path.exists(), f"gclf_file_path [{str(gclf_file_path)}] does not exist."
    lidar_log_path = Path(lidar_log_path)
    assert lidar_log_path.exists(), f"lidar_log_path [{str(lidar_log_path)}] does not exist."

    LIDAR_CAL = readXmlExtrinsicCal(lidar_cal_path)[lidar_id]
    CAMERA_CAL = readXmlCameraCal(camera_cal_path)[camera_id]

    T_BODY_CAM = pt.transform_from_pq(CAMERA_CAL.pq_T_imu_x)
    T_CAM_BODY = np.linalg.inv(T_BODY_CAM)
    K_CAM_mtx = RRU._K_arr_to_mtx(CAMERA_CAL.K)

    # Spawn a new recording
    rr.init("VisSyncedLidarCamera", spawn=True, recording_id=str(time.perf_counter_ns()))
    rr.log("/", rr.ViewCoordinates.FLU, static=True)
    rr.log(WORLD_FRAME, RRU._T_to_rerun(np.eye(4)), static=True)
    rr.log(BODY_FRAME, RRU._T_to_rerun(np.eye(4)))
    rr.log(BODY_FRAME, RRU._create_origin_axes(0.5), static=True)
    rr.log(LIDAR_FRAME, RRU._T_to_rerun(
        pt.transform_from_pq(LIDAR_CAL.pq_T_imu_x)), static=True)
    rr.log(LIDAR_FRAME, RRU._create_origin_axes(0.1), static=True)
    rr.log(CAMERA_FRAME, RRU._T_to_rerun(
        pt.transform_from_pq(CAMERA_CAL.pq_T_imu_x)), static=True)
    rr.log(CAMERA_FRAME, rr.Pinhole(
        image_from_camera=RRU._K_arr_to_mtx(CAMERA_CAL.K),
        height=CAMERA_CAL.hw[0], width=CAMERA_CAL.hw[1],
        image_plane_distance=0.25),
        static=True)

    # Cumulative variables for visualisation
    cumulative_path: list[np.ndarray] = []

    gclfFile = reader.GenericCameraReader(str(gclf_file_path))
    scanFile = ScanFile(lidar_log_path)

    prev_t = -1

    for t_ns, T_world_body, pc_body in tqdm(scanFile, desc="logged_point_cloud"):
        if prev_t > 0:
            dt_ns = t_ns - prev_t
            dt_ms = dt_ns / 1e6
            if dt_ms > 100.0:
                print(f"\n[WARN] Frame gap: {dt_ms:.2f}ms!")

        prev_t = t_ns

        gclfFile.seek_to_timestamp(t_ns)
        image_pil, _ = gclfFile.get_next()
        image = np.array(image_pil)

        # Preprocessing
        pc_cam_frame = PCP.transform_pc(pc_body, T_CAM_BODY)
        depth_img = PCP.get_pc_as_depth_img(pc_cam_frame, K_CAM_mtx, CAMERA_CAL.hw)

        # Visualise things
        rr.set_time(TIMELINE, timestamp=np.datetime64(t_ns, 'ns'))

        rr.log(BODY_FRAME, RRU._T_to_rerun(T_world_body))
        rr.log(IMAGE_FRAME, rr.Image(image).compress(jpeg_quality=90))
        rr.log(DEPTH_FRAME, rr.DepthImage(depth_img, depth_range=(0.1, 10.0)))
        cumulative_path = RRU.log_position(T_world_body, cumulative_path)
        rr.log(f"{BODY_FRAME}/curr_scan", rr.Points3D(
            pc_body, colors=RRU.color_pc(
                pc_body, cmap='summer',
                func=lambda x: np.linalg.norm(x, axis=1))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise synced LiDAR and camera data in Rerun.")
    parser.add_argument("--log_folder", type=Path, required=True,
                        help="Path to the folder with log files.")
    parser.add_argument("--camera_cal_path", type=Path,
                        default=Path("CameraCalRaiboAsQUGV113.xml"),
                        help="Path to camera calibration XML.")
    parser.add_argument("--lidar_cal_path", type=Path,
                        default=Path("ExtrinsicCalRaiboAsQUGV113.xml"),
                        help="Path to LiDAR calibration XML.")
    parser.add_argument("--lidar_id", type=int, default=71,
                        help="LiDAR device ID (default: 71).")
    parser.add_argument("--camera_id", type=int, default=211,
                        help="Camera device ID (default: 211).")
    args = parser.parse_args()

    gclf_file_path = Path(args.log_folder) / f"camera{args.camera_id}.gclf"
    lidar_log_path = Path(args.log_folder) / "PoseAndPointClouds.bin"

    main(camera_cal_path=args.camera_cal_path,
         lidar_cal_path=args.lidar_cal_path,
         gclf_file_path=gclf_file_path,
         lidar_log_path=lidar_log_path,
         lidar_id=args.lidar_id,
         camera_id=args.camera_id)
