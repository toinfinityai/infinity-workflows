import cv2
import imageio
import logging

from typing import Optional
from IPython.display import Image, display


class IgnoreBlackWarning(logging.Filter):
    def filter(self, record):
        return "Black is not installed" not in record.msg


def suppress_useless_warnings():
    logging.getLogger("papermill.translators").addFilter(IgnoreBlackWarning())


def display_image(image_path: str, display_width: int = 600):
    display(Image(data=open(image_path, "rb").read(), format="png", width=display_width))


def display_video_as_gif(
    video_path: str,
    output_path: Optional[str] = None,
    downsample_resolution: int = 1,
    downsample_frames: int = 1,
    display_width: int = 600,
):
    """Displays a video as a gif, using the display feature native to notebooks.
    This method will also allow for persistent caching of the gif for easy (and performant) viewing.
    Args:
        video_path: Path to the video to display.
        output_path: Path to the output gif.
        downsample_resolution: Downsample the video to this resolution (e.g. downsample_resolution=2 will downsample the video to 1/2 the resolution).
        downsample_frames: Downsample the video to this number of frames (e.g. downsample_frames=2 will downsample the video to 1/2 the number of frames).
        display_width: Width of the image to display.
    """

    if output_path is None:
        output_path = video_path.replace(".mp4", ".gif")

    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_index = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if frame_index % downsample_frames == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame[::downsample_resolution, ::downsample_resolution, :]
            frames.append(frame)
        frame_index += 1
    imageio.mimsave(output_path, frames, format="GIF", duration=1 / fps * downsample_frames)
    display_image(output_path, display_width)
