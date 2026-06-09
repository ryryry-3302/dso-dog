import os

import numpy as np
import struct
import pytransform3d.transformations as pt
import pytransform3d.rotations as pr

"""
**MLOAMC Binary Scan Format**
The logging format is a sequence of Scans, where a Scan is:

Scans are expressed in the *Body Frame* of the robot.

ScanHeader,
Sequence of ScanPoints

Where ScanHeader is:
Timestamp (100s of ns): int64
Pose: Pose
numPoints: int32

and Pose is:
x, y, z, roll, pitch, yaw: float64

and ScanPoint is:
x, y, z: float32
intensity: uint8

Little endian, no padding.
It is possible for a ScanPoint to be xcm==0, ycm==0, zcm==0, which means it is a lidar beam without return.

Note that, where the relative motion 
"""


class ScanFile:
    """
    Decode a binary scan file.

    Each Scan: ScanHeader + ScanPoints
    ScanHeader: timestamp (100s of ns) (int64), pose (6*float64), numPoints (int32)
    ScanPoint: x,y,z (float32), intensity (uint8)
    Little-endian, no padding.
    """

    # '<' - Little Endian
    # q: int64 (signed long long)
    # 6d : 6 doubles
    # i: int32
    _header_fmt = '<q6di'      # timestamp, pose, numPoints
    _header_size = struct.calcsize(_header_fmt)
    _point_dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                             ('intensity', '<u1')])
    _point_size = 13  # 4+4+4+1

    def __init__(self, filename):
        self.timestamp_to_seek_pos: dict[int, int] = {}
        self._f = open(filename, 'rb')
        self._n_scans = 0
        f = self._f
        while True:
            curr_pos = self._f.tell()
            header = f.read(self._header_size)
            if len(header) < self._header_size:
                break
            t_100ns, tx, ty, tz, rx, ry, rz, num = \
                struct.unpack(self._header_fmt, header)
            t_ns = t_100ns * 100
            f.seek(num * self._point_size, os.SEEK_CUR)          # skip points
            self._n_scans += 1
            self.timestamp_to_seek_pos[t_ns] = curr_pos

        self._timestamps_ns: np.ndarray = np.array(
            [t for t in self.timestamp_to_seek_pos.keys()], dtype=int
        )

        # Reset to the start for further reading
        self._f.seek(0)

    def __len__(self):
        return self._n_scans

    def __iter__(self):
        self._f.seek(0)
        return self

    def __next__(self):
        '''
        timestamp_ns, T_W_C(4x4), pts[N, 3]
        '''
        header = self._f.read(self._header_size)
        if len(header) < self._header_size:
            raise StopIteration
        timestamp_100ns, tx, ty, tz, roll, pitch, yaw, num = struct.unpack(
            self._header_fmt, header)
        t_ns = timestamp_100ns * 100
        pts = np.frombuffer(self._f.read(
            num * self._point_size), dtype=self._point_dtype)
        # filter no‑return points
        mask = (pts['x'] != 0) | (pts['y'] != 0) | (pts['z'] != 0)
        pts = pts[mask]
        cloud = np.vstack((pts['x'], pts['y'], pts['z'])).T.astype(np.float64)

        T = np.eye(4, dtype=np.float64)
        R = pr.matrix_from_euler([roll, pitch, yaw], 0, 1, 2, extrinsic=True)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]

        return t_ns, T, cloud

    def seek_to_time(self, t_ns: int):
        # We have the exact timestamp
        if t_ns in self.timestamp_to_seek_pos:
            self._f.seek(self.timestamp_to_seek_pos[t_ns])
            return self.__next__()

        # We don't have the exact timestamp, look for the closest one (and
        # optionally print out t_delta)
        closest_idx = np.argmin(np.abs(t_ns - self._timestamps_ns))
        closest_key = self._timestamps_ns[closest_idx]
        dt_ns = closest_key - t_ns
        print(f"dt between query t_ns and lidar timestamp: {dt_ns / 1e6:.3f}ms")

        self._f.seek(self.timestamp_to_seek_pos[closest_key])
        return self.__next__()
