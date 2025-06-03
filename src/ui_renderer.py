import cv2
import numpy as np


def draw_ui(live_frame, label, state, sentence):
    target_height = 720
    video_aspect_ratio = live_frame.shape[1] / live_frame.shape[0]

    video_height = target_height
    video_width = int(video_height * video_aspect_ratio)
    resized_video = cv2.resize(live_frame, (video_width, video_height))

    text_area_width = 400
    screen_width = video_width + text_area_width

    canvas = np.full((video_height, screen_width, 3), (230, 250, 250), dtype=np.uint8)

    canvas[:, :video_width] = resized_video

    text_x = video_width + 20

    cv2.putText(
        canvas,
        f"Live: {label}",
        (text_x, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 100, 255),
        2,
    )
    cv2.putText(
        canvas,
        f"State: {state}",
        (text_x, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 180, 0),
        2,
    )

    separator_y = 120  
    cv2.line(
        canvas,
        (text_x, separator_y),
        (text_x + text_area_width - 40, separator_y),
        (160, 160, 160),
        2,
    )

    draw_multiline_text(
        image=canvas,
        text=sentence,
        position=(text_x, 160),
        font=cv2.FONT_HERSHEY_SIMPLEX,
        scale=0.8,
        color=(60, 60, 60),
        thickness=2,
        max_width=text_area_width - 40,
    )

    return canvas


def draw_multiline_text(
    image, text, position, font, scale, color, thickness, max_width
):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + " " + word if current_line else word
        (text_size, _) = cv2.getTextSize(test_line, font, scale, thickness)
        if text_size[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    x, y = position
    line_height = int(cv2.getTextSize("Test", font, scale, thickness)[0][1] * 1.5)

    for i, line in enumerate(lines):
        cv2.putText(
            image, line, (x, y + i * line_height), font, scale, color, thickness
        )