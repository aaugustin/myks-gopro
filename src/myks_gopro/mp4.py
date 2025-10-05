"""
Read MP4 files.

Requires ffmpeg and ffprobe.

"""

from __future__ import annotations

import functools
import json
import pathlib
import subprocess
import sys
from decimal import Decimal
from fractions import Fraction
from typing import Any, cast


__all__ = ["MP4File"]


class MP4File:
    def __init__(self, mp4_path: str | pathlib.Path):
        self.mp4_path = mp4_path

    @functools.cache
    def get_streams(self) -> list[dict[str, Any]]:
        return cast(
            list[dict[str, Any]],
            json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-hide_banner",
                        "-output_format",
                        "json",
                        "-show_streams",
                        self.mp4_path,
                    ],
                    stderr=subprocess.DEVNULL,
                )
            )["streams"],
        )

    @functools.cached_property
    def video_metadata(self) -> dict[str, Any]:
        streams = [
            stream for stream in self.get_streams() if stream["codec_type"] == "video"
        ]
        if len(streams) != 1:
            raise ValueError(f"expected 1 video stream, found {len(streams)}")
        stream = streams[0]

        frame_rate = Fraction(stream["r_frame_rate"])
        time_base = Fraction(stream["time_base"])
        duration_ts = int(stream["duration_ts"])
        duration = Decimal(stream["duration"])
        nb_frames = int(stream["nb_frames"])

        assert duration == Decimal(float(time_base * duration_ts)).quantize(duration)
        assert nb_frames == frame_rate * time_base * duration_ts

        return {
            "index": int(stream["index"]),
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "frame_rate": frame_rate,
            "time_base": time_base,
            "duration_ts": duration_ts,
            "duration": duration,
            "nb_frames": nb_frames,
        }

    @functools.cached_property
    def gpmd_metadata(self) -> dict[str, Any]:
        streams = [
            stream
            for stream in self.get_streams()
            if stream["codec_type"] == "data" and stream["codec_tag_string"] == "gpmd"
        ]

        if len(streams) != 1:
            raise ValueError(f"expected 1 gpmd stream, found {len(streams)}")
        stream = streams[0]

        time_base = Fraction(stream["time_base"])
        duration_ts = int(stream["duration_ts"])
        duration = Decimal(stream["duration"])
        nb_frames = int(stream["nb_frames"])

        assert duration == Decimal(float(time_base * duration_ts)).quantize(duration)

        return {
            "index": int(stream["index"]),
            "time_base": time_base,
            "duration_ts": duration_ts,
            "duration": duration,
            "nb_frames": nb_frames,
        }

    def extract_gpmf(self) -> bytes:
        """Extract the GPMF stream."""
        return subprocess.check_output(
            [
                "ffmpeg",
                "-i",
                self.mp4_path,
                "-map",
                f"0:{self.gpmd_metadata['index']}",
                "-f",
                "rawvideo",
                "-",
            ],
            stderr=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} video.mp4 video.gpmf\n")
        sys.exit(2)

    mp4_path = pathlib.Path(sys.argv[1])
    gpmf_file = pathlib.Path(sys.argv[2])
    gpmf_file.write_bytes(MP4File(mp4_path).extract_gpmf())
