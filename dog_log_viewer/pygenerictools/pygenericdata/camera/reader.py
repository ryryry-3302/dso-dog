import os
import struct

import numpy as np
from PIL import Image as image_module
from PIL.Image import Image
from pygenericdata import utils

_CHUNK_HEADER_SIZE = 8
_LOG_HEADER_CHUNK_PAYLOAD_SIZE = 12

_LOG_HEADER_MAGIC = "464c4347"
_IMG_HEADER_MAGIC = "52444849"
_IMG_DATA_MAGIC = "54414449"

IMG_HEADER_PAYLOAD_FORMATS = {
    "timestamp": "q",
    "sensorId": "L",
    "height": "L",
    "width": "L",
    "encoding": "L",
    "seq": "Q",
    "exposureTime": "l",
    "gain": "l",
    "attributeCount": "L",
    "attributeTypes": "L",
    "attributeValues": "l",
}

FORMAT_TO_SIZE = {"Q": 8, "q": 8, "L": 4, "l": 4}

IMG_HEADER_PAYLOAD_SIZES = {
    k: FORMAT_TO_SIZE[v] for k, v in IMG_HEADER_PAYLOAD_FORMATS.items()
}
VARIABLE_IMG_HEADER_PAYLOADS = ["attributeTypes", "attributeValues"]


def _little_endian_format(format_codes):
    return "<" + "".join(format_codes)


class GenericCameraReader:
    def __init__(self, gclf_file: str):
        assert (
            os.path.exists(gclf_file) and gclf_file[-5:] == ".gclf"
        ), "{} cannot be read".format(gclf_file)
        self.gclf_file = gclf_file

        self.gclf_handle = open(self.gclf_file, "rb")
        self.source_segment_id = -1
        self.index_file_position = -1
        self._read_file_header()

        self._curr_img_metadata = {}
        self._curr_img = None

        # Build a timestamp-to-bytes map for faster seeking
        self._timestamp_to_bytes_map: dict[int, int] = {}

    def _read_file_header(self):
        chunk_type, payload_size = self._read_chunk_header()
        if chunk_type != _LOG_HEADER_MAGIC:
            raise IOError(
                "ERR: expected file header chunk type: {}; actual: {}".format(
                    _LOG_HEADER_MAGIC, chunk_type
                )
            )
        if payload_size != _LOG_HEADER_CHUNK_PAYLOAD_SIZE:
            raise IOError(
                "ERR: expected file header payload size: {}; actual: {}".format(
                    _LOG_HEADER_CHUNK_PAYLOAD_SIZE, payload_size
                )
            )
        payload = self._read_bytes(payload_size)
        self.source_segment_id = utils.unpack_u32_le(payload[:4])
        self.index_file_position = utils.unpack_u64_le(payload[4:])

    def _read_chunk_header(self):
        data = self._read_bytes(_CHUNK_HEADER_SIZE)
        chunk_type = data[:4]
        chunk_type.reverse()
        chunk_type = chunk_type.hex()
        payload_size = utils.unpack_u32_le(data=data[4:])
        return chunk_type, payload_size

    def _read_bytes(self, num_bytes):
        data = bytearray(self.gclf_handle.read(num_bytes))
        if len(data) != num_bytes:
            raise IOError(
                "ERR: # bytes read:\n\texpected: {}; actual : {}".format(
                    num_bytes, len(data)
                )
            )
        return data

    def __del__(self):
        self.gclf_handle.close()

    def _read_img_header(self):
        try:
            chunk_type, payload_size = self._read_chunk_header()
            chunk_header_start = self.gclf_handle.tell() - _CHUNK_HEADER_SIZE
        except IOError:
            raise StopIteration("end of image sequence")

        if chunk_type != _IMG_HEADER_MAGIC:
            raise IOError(
                "ERR: expected img header chunk type: {}; actual: {}".format(
                    _IMG_HEADER_MAGIC, chunk_type
                )
            )
        attributes = list(IMG_HEADER_PAYLOAD_FORMATS.keys())
        fixed_size_attributes = [
            item
            for item in attributes
            if item not in ["attributeTypes", "attributeValues"]
        ]
        payload = self._read_bytes(payload_size)

        buffer_format = _little_endian_format(
            [IMG_HEADER_PAYLOAD_FORMATS[k] for k in fixed_size_attributes]
        )
        buffer_size = sum([IMG_HEADER_PAYLOAD_SIZES[k]
                          for k in fixed_size_attributes])
        buffer_vals = list(struct.unpack(buffer_format, payload[:buffer_size]))
        metadata = {
            attrib: buffer_vals[i] for i, attrib in enumerate(fixed_size_attributes)
        }

        payload = payload[buffer_size:]
        attribute_count = metadata["attributeCount"]
        if attribute_count == 0:
            metadata["attributeTypes"] = list()
            metadata["attributeValues"] = list()
            assert len(payload) == 0
        else:
            buffer_format = _little_endian_format(
                [IMG_HEADER_PAYLOAD_FORMATS["attributeTypes"]] * attribute_count
                + [IMG_HEADER_PAYLOAD_FORMATS["attributeValues"]] * attribute_count
            )
            buffer_size = attribute_count * sum(
                [
                    IMG_HEADER_PAYLOAD_SIZES[k]
                    for k in ["attributeTypes", "attributeValues"]
                ]
            )
            assert len(payload) == buffer_size
            buffer_vals = list(struct.unpack(buffer_format, payload))
            metadata["attributeTypes"] = buffer_vals[:attribute_count]
            metadata["attributeValues"] = buffer_vals[
                attribute_count: attribute_count * 2
            ]
        self._curr_img_metadata = metadata

        self._timestamp_to_bytes_map[metadata["timestamp"]
                                     ] = chunk_header_start

    def _read_image(self) -> tuple[Image, dict]:
        """
        Returns:
            * PIL Image
            * metadata of image e.g. timestamps, gain etc.
        """
        chunk_type, payload_size = self._read_chunk_header()
        if chunk_type != _IMG_DATA_MAGIC:
            raise IOError(
                "ERR: expected img header chunk type: {}; actual: {}".format(
                    _LOG_HEADER_MAGIC, chunk_type
                )
            )
        payload = self._read_bytes(num_bytes=payload_size)
        encoding = self._curr_img_metadata["encoding"]
        height = self._curr_img_metadata["height"]
        width = self._curr_img_metadata["width"]
        if encoding == 1:
            num_bytes = height * width * 3
            assert payload_size == num_bytes
            image = list(struct.unpack("B" * num_bytes, payload))
            image = np.array(image).astype(np.uint8).reshape(height, width, 3)
            image = image_module.fromarray(image)

        elif encoding == 2:
            num_bytes = height * width
            assert payload_size == num_bytes
            image = list(struct.unpack("B" * num_bytes, payload))
            image = np.array(image).astype(np.uint8).reshape(height, width)
            image = image_module.fromarray(image)
        elif encoding == 4:  # MONO16
            num_bytes = height * width * 2
            assert payload_size == num_bytes
            image = list(struct.unpack("H" * (num_bytes // 2), payload))
            image = np.array(image).astype(np.uint16).reshape(height, width)
            image = image_module.fromarray(image)
        else:
            raise NotImplementedError
        self._curr_img = image
        return self._curr_img, self._curr_img_metadata

    def get_next(self):
        self._read_img_header()
        return self._read_image()

    def seek_to_timestamp(self, timestamp_ns: int):
        '''
        Move the file handle to the start of the first Image Header Chunk with timestamp strictly
        greater than timestamp_ns. Raise StopIteration if the timestamp is at end of file.
        '''

        # We've found it before.
        if timestamp_ns in self._timestamp_to_bytes_map:
            # print(f"Found {timestamp_ns} before, skipping directly...")
            self.gclf_handle.seek(self._timestamp_to_bytes_map[timestamp_ns])
            return

        # Haven't found it yet, need to seek from the beginning
        # print("[seek_to_timestamp]: At: ", self.gclf_handle.tell())
        self.gclf_handle.seek(
            _LOG_HEADER_CHUNK_PAYLOAD_SIZE + _CHUNK_HEADER_SIZE)
        # print("[seek_to_timestamp]: After seeking first, at: ", self.gclf_handle.tell())

        while True:
            chunk_start_pos = self.gclf_handle.tell()
            # print("[seek_to_timestamp]: Before reading image header: ", chunk_start_pos)
            # Read header. This will also raise StopIteration, if we are at EOF.
            self._read_img_header()
            # print("[seek_to_timestamp]: After reading image header: ", chunk_start_pos)

            curr_t_ns = self._curr_img_metadata["timestamp"]
            self._timestamp_to_bytes_map[curr_t_ns] = chunk_start_pos
            # print("  Curr timestamp:", curr_t_ns, "desired timestamp: ", timestamp_ns,
            #       "diff:", timestamp_ns - curr_t_ns)

            # Check timestamp
            if curr_t_ns >= timestamp_ns:
                # Restore to the header's position, and break.
                self.gclf_handle.seek(chunk_start_pos)
                # print("Found it!")
                break

            # If not, seek over the image.
            chunk_type, payload_size = self._read_chunk_header()
            # print("[seek_to_timestamp]: After reading image data header: ", chunk_start_pos)

            if chunk_type != _IMG_DATA_MAGIC:
                raise IOError(
                    f"ERR: expected img data chunk type: {_IMG_DATA_MAGIC}; actual: {chunk_type}")

            # print("[seek_to_timestamp]: Image payload size: ", payload_size)
            self.gclf_handle.seek(payload_size, os.SEEK_CUR)
