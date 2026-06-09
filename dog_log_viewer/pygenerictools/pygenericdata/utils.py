import numpy as np
import struct


def unpack_u64_le(data: bytearray):
    assert len(data) == 8
    out = struct.unpack('<Q', data)[0]
    return out


def unpack_i64_le(data: bytearray):
    assert len(data) == 8
    out = struct.unpack('<q', data)[0]
    return out


def unpack_u32_le(data: bytearray):
    assert len(data) == 4
    out = struct.unpack('<L', data)[0]
    return out


def unpack_i32_le(data: list):
    assert len(data) == 4
    out = struct.unpack('<l', data)[0]
    return out


def unpack_float_le(data: bytearray):
    assert len(data) == 4
    out = struct.unpack('<f', data)[0]
    return out


def unpack_u16_le(data: bytearray):
    assert len(data) == 2
    out = struct.unpack('<H', data)[0]
    return out

