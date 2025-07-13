import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def get_fourcc(extension):
    ext = extension.lower()
    if ext == '.mp4':
        return cv2.VideoWriter_fourcc(*'mp4v')
    elif ext == '.avi':
        return cv2.VideoWriter_fourcc(*'XVID')
    elif ext == '.mov':
        return cv2.VideoWriter_fourcc(*'MJPG')
    else:
        raise ValueError(f"Unsupported video format: {ext}")

def save_video(output_video_frames, output_video_path):
    if not output_video_frames:
        raise ValueError("No frames to save.")
    extension = os.path.splitext(output_video_path)[-1]
    fourcc = get_fourcc(extension)
    height, width = output_video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
