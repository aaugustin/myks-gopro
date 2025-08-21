"""
Read MP4 files.

Requires ffmpeg and ffprobe.

"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import functools
import json
import pathlib
import subprocess
import sys
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

        if nb_frames != frame_rate * time_base * duration_ts:
            raise ValueError(
                f"expected {nb_frames} frames, "
                f"got {frame_rate * time_base * duration_ts}"
            )
        assert duration == Decimal(float(time_base * duration_ts)).quantize(duration)

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

    def check_gpmf_packets_sequence(self) -> None:
        """Check that the sequence of GPMF packets is regular."""
        video_meta = self.video_metadata
        gpmd_meta = self.gpmd_metadata

        packets = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-hide_banner",
                    "-output_format",
                    "json",
                    "-select_streams",
                    str(gpmd_meta["index"]),
                    "-show_packets",
                    self.mp4_path,
                ],
                stderr=subprocess.DEVNULL,
            )
        )["packets"]

        if len(packets) != gpmd_meta["nb_frames"]:
            raise ValueError(
                f"expected one packet per frame, "
                f"got {len(packets)} for {gpmd_meta['nb_frames']} frames"
            )

        packet = packets[0]
        ts = int(packet["pts"])
        duration = int(packet["duration"])

        for index, packet in enumerate(packets[:-1]):  # last packet may be shorter
            if int(packet["duration"]) != duration:
                raise ValueError(
                    f"expected constant packet duration {duration}, "
                    f"got {int(packet['duration'])} at frame {index}"
                )

        for index, packet in enumerate(packets):
            if int(packet["pts"]) != ts:
                raise ValueError(
                    f"expected packet timestamps {ts}, "
                    f"got {int(packet['pts'])} at frame {index}"
                )
            ts += duration

        if (
            abs(gpmd_meta["duration"] - video_meta["duration"])
            > float(max(video_meta["time_base"], gpmd_meta["time_base"])) / 2
        ):
            raise ValueError(
                f"expected video and gpmd durations to be almost equal, "
                f"got {gpmd_meta['duration']} and {video_meta['duration']}"
            )

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
