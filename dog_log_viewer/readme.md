# Playback Tools for EAI Logs

This repository provides tools to visualize synchronized LiDAR and camera data from EAI (Embodied AI) robot logs.

## Quick Start

### Prerequisites

- Make a venv (you should consider using `uv`), and install from `requirements.txt`.
- List might not be complete, help each other out here if something goes wrong.

### Running the Visualizer

The main entry point is `vis_and_read_log.py`, which streams synced LiDAR point clouds and camera images into [Rerun](https://rerun.io/).

```bash
python vis_and_read_log.py --log_folder <path_to_log_folder>
```

#### Argparse Flags

| Flag                | Default                          | Description                                                                                                                                                                |
| ------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--log_folder`      | **(required)**                   | Directory containing the log files. Expected contents:<br>• `camera<CAMERA_ID>.gclf` — camera image stream<br>• `PoseAndPointClouds.bin` — LiDAR pose + point cloud stream |
| `--camera_cal_path` | `CameraCalRaiboAsQUGV113.xml`    | Path to the **camera calibration XML** file.                                                                                                                               |
| `--lidar_cal_path`  | `ExtrinsicCalRaiboAsQUGV113.xml` | Path to the **LiDAR extrinsic calibration XML** file.                                                                                                                      |
| `--lidar_id`        | `71`                             | LiDAR device ID to select from the calibration file.                                                                                                                       |
| `--camera_id`       | `211`                            | Camera device ID to select from the calibration file.                                                                                                                      |

#### Example

```bash
python vis_and_read_log.py \
  --log_folder ./logs/run_001 \
  --camera_cal_path ./calib/CameraCalRaiboAsQUGV113.xml \
  --lidar_cal_path ./calib/ExtrinsicCalRaiboAsQUGV113.xml \
  --lidar_id 71 \
  --camera_id 211
```

## Calibration Files

### What is an Extrinsic Calibration?

> **NOTE:** Feel free to read this summary from Kimi-k2.6; otherwise ask Tian Yi for a more detailed explanation.

**Extrinsic calibration** defines the rigid-body transform (rotation + translation) that converts coordinates from a *sensor frame* to the *body (IMU) frame*. This can be denoted `T_body_sensor`.

In this codebase:
- **LiDAR extrinsics** (`--lidar_cal_path`) describe where the LiDAR is mounted on the robot relative to the IMU. Each entry contains `imuToLidarOffset{X,Y,Z}` and `imuToLidarQ{w,x,y,z}` — i.e., the pose of the LiDAR expressed in the body frame.
- When the code visualizes data, it uses this transform to show the LiDAR point cloud in the correct place on the robot model.

### What is a Camera Calibration?

**Camera calibration** has two parts:

1. **Intrinsic calibration** — how the camera projects 3-D world points onto its 2-D image sensor.
   - `K = (fu, fv, cu, cv)`: focal lengths and principal point.
   - `D = (k1, k2, p1, p2)`: radial and tangential distortion coefficients.
   - `xi`: parameter used for some wide-angle / fisheye models. For the pinhole cams we logged (111, 211) these are always set to 0.
   - `hw = (height, width)`: image resolution.

2. **Extrinsic component** — the camera's pose relative to the IMU body frame, stored as `imuToCameraOffset{X,Y,Z}` and `imuToCameraQ{w,x,y,z}`. This lets the code transform the LiDAR point cloud into the camera frame so that a depth image can be rendered and overlayed.

Together, the camera calibration file tells the visualizer both *how the lens works* and *where the camera is on the robot*.

## Data Formats

- **LiDAR**: `PoseAndPointClouds.bin` — a custom binary stream of scans. Each scan has a header (timestamp, 6-DOF pose, point count) followed by point data (`x, y, z` as `float32`, intensity as `uint8`). See `pygenerictools/readLidarBinaryFormat.py` for the decoder.
- **Camera**: `.gclf` — a chunked binary log of images with per-frame metadata (timestamp, exposure, gain, etc.). See `pygenerictools/pygenericdata/camera/reader.py` for the decoder.
