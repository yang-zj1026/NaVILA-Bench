from typing import List, Optional

import cv2
import attr
import textwrap
import gzip
import json
import numpy as np


@attr.s(auto_attribs=True)
class InstructionData:
    instruction_text: str
    instruction_tokens: Optional[List[str]] = None


def skip(*args, **kwargs):
    pass


def read_episodes(file_path):
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    
    return data["episodes"]


def add_instruction_on_img(img: np.ndarray, text: str, start_y=0) -> None:
    font_size = 0.6
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    char_size = cv2.getTextSize(" ", font, font_size, thickness)[0]
    wrapped_text = textwrap.wrap(
        text, width=int((img.shape[1] - 15) / char_size[0])
    )
    if len(wrapped_text) < 8:
        wrapped_text.insert(0, "")

    y = start_y
    start_x = 15
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, thickness)[0]
        y += textsize[1] + 25
        cv2.putText(
            img,
            line,
            (start_x, y),
            font,
            font_size,
            (0, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )


def get_vel_command(text):
    if "turn left" in text.lower():
        if "45" in text.lower():
            return [0.0, 0.0, np.pi/6.0], 1.5
        elif "30" in text.lower():
            return [0.0, 0.0, np.pi/6.0], 1.0
        elif "15" in text.lower():
            return [0.0, 0.0, np.pi/6.0], 0.5
        return [0.0, 0.0, np.pi/6.0], 0.5
    elif "turn right" in text.lower():
        if "45" in text.lower():
            return [0.0, 0.0, -np.pi/6.0], 1.5
        elif "30" in text.lower():
            return [0.0, 0.0, -np.pi/6.0], 1.0
        elif "15" in text.lower():
            return [0.0, 0.0, -np.pi/6.0], 0.5
        return [0.0, 0.0, -np.pi/6.0], 0.5
    elif "move forward" in text.lower() or "move" in text.lower():
        if "75" in text.lower():
            return [0.5, 0.0, 0.0], 1.5
        elif "50" in text.lower():
            return [0.5, 0.0, 0.0], 1.0
        elif "25" in text.lower():
            return [0.5, 0.0, 0.0], 0.5
        return [0.5, 0.0, 0.0], 0.5
    elif "stop" in text.lower():
        return [0.0, 0.0, 0.0], 0.0
    else:
        return [0.5, 0.0, 0.0], 0.5