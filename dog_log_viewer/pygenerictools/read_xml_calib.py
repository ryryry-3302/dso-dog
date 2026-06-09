import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Union, NamedTuple


class CameraCalibration(NamedTuple):
    xi: float
    K: tuple[float, float, float, float]
    D: tuple[float, float, float, float]
    pq_T_imu_x: list[float]
    hw: tuple[int, int]


class LidarCalibration(NamedTuple):
    numBeams: int
    numAzimuthBins: int
    vfovUp: float
    vfovDown: float
    pq_T_imu_x: list[float]


def _convert_text(text: str) -> Union[str, int, float]:
    """
    Try to interpret a string as int/float, falling back to str.
    Empty string stays empty.
    """
    if text == '':
        return ''
    try:
        # Try int first (e.g., sensorId)
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text  # keep as string


def _parse_element(elem: ET.Element):
    """
    Recursively turn an ElementTree node into a dict/list/value.

    Returns:
        - If the element has no child elements: its text (converted to int/float
          if possible, or empty string).
        - If the element has children:
            * If a tag occurs only once: a dict mapping tag → child.
            * If a tag occurs multiple times: a list of dicts.
    """
    # Strip whitespace from the element's text; keep None as None
    text = elem.text.strip() if elem.text else None

    # If the element has no children, return its (converted) text
    if len(elem) == 0:
        if text is None:
            return None
        else:
            return _convert_text(text)

    # Otherwise, process children
    child_map: dict[str, list[Any]] = defaultdict(list)

    for child in elem:
        child_map[child.tag].append(_parse_element(child))

    # Build the resulting structure
    result: dict[str, Any] = {}
    for tag, values in child_map.items():
        if len(values) == 1:
            result[tag] = values[0]
        else:
            result[tag] = values   # list of dicts / values

    # If the element itself contains useful text (e.g., attributes or a mix)
    if text is not None and text != "":
        # Put the text under a special key – adjust as needed
        result['_text'] = _convert_text(text)

    return result


def xml_to_dict(filepath: Path | str) -> Any:
    """
    Does what it says: Parse an XML file into a dictionary.
    """
    return _parse_element(ET.parse(filepath).getroot())


def parse_camera_dict(cam_dict: dict) -> CameraCalibration:
    '''
    Parse a dict that looks like this
    {'deviceName': 'rightRaw',
    'distortion': {'k1': -0.11812,'k2': 0.995257,'p1': -0.003124,'p2': -0.000425},
    'imageHeight': 480,
    'imageWidth': 640,
    'imuToCameraOffsetX': 0.064918,
    'imuToCameraOffsetY': -0.153975,
    'imuToCameraOffsetZ': 0.002419,
    'imuToCameraQw': 0.512544,
    'imuToCameraQx': -0.482425,
    'imuToCameraQy': 0.52616,
    'imuToCameraQz': -0.4772,
    'maskFile': None,
    'projection': {'cu': 320.517626,'cv': 246.633654,'fu': 974.619158,'fv': 971.293065,'xi': 2.254507},
    'sensorId': 0}
    '''

    proj = cam_dict['projection']
    dist = cam_dict['distortion']
    xi = proj['xi']
    K = (proj['fu'], proj['fv'], proj['cu'], proj['cv'])
    D = (dist['k1'], dist['k2'], dist['p1'], dist['p2'])
    pq = [cam_dict['imuToCameraOffsetX'],
          cam_dict['imuToCameraOffsetY'],
          cam_dict['imuToCameraOffsetZ'],
          cam_dict['imuToCameraQw'],
          cam_dict['imuToCameraQx'],
          cam_dict['imuToCameraQy'],
          cam_dict['imuToCameraQz']]
    hw = (cam_dict['imageHeight'], cam_dict['imageWidth'])

    return CameraCalibration(xi, K, D, pq, hw)


def parse_extrinsic_dict(extrinsic_dict: dict) -> LidarCalibration:
    '''
    Parse a dict that looks like this:
    {'imuToLidarOffsetX': 0.0,
    'imuToLidarOffsetY': 0,
    'imuToLidarOffsetZ': 0.26,
    'imuToLidarQw': 1.0,
    'imuToLidarQx': 0.0,
    'imuToLidarQy': 0.0,
    'imuToLidarQz': 0.0,
    'lidarMaskPath': None,
    'numAzimuthBins': 2048,
    'numBeams': 64,
    'sensorId': 71,
    'vfovDown': 45,
    'vfovUp': -45}
    '''
    numBeams = extrinsic_dict['numBeams']
    numAzimuthBins = extrinsic_dict['numAzimuthBins']
    vfovUp = extrinsic_dict['vfovUp']
    vfovDown = extrinsic_dict['vfovDown']
    pq = [extrinsic_dict['imuToLidarOffsetX'],
          extrinsic_dict['imuToLidarOffsetY'],
          extrinsic_dict['imuToLidarOffsetZ'],
          extrinsic_dict['imuToLidarQw'],
          extrinsic_dict['imuToLidarQx'],
          extrinsic_dict['imuToLidarQy'],
          extrinsic_dict['imuToLidarQz']]

    return LidarCalibration(numBeams, numAzimuthBins, vfovUp, vfovDown, pq)


def readXmlCameraCal(cam_cal_path: str | Path) -> dict[int, CameraCalibration]:
    calib_dict = xml_to_dict(cam_cal_path)['cameraCalibration']
    if isinstance(calib_dict, list):
        cameras = {elem['sensorId']: parse_camera_dict(
            elem) for elem in calib_dict}
    else:
        cameras = {calib_dict['sensorId']: parse_camera_dict(calib_dict)}
    return cameras


def readXmlExtrinsicCal(cal_path: str | Path) -> dict[int, LidarCalibration]:
    calib_dict = xml_to_dict(cal_path)['lidarCalibration']
    if isinstance(calib_dict, list):
        lidars = {elem['sensorId']: parse_extrinsic_dict(
            elem) for elem in calib_dict}
    else:
        lidars = {calib_dict['sensorId']: parse_extrinsic_dict(calib_dict)}
    return lidars
