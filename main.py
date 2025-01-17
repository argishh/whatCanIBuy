import os

from utils import (detect_shoppable_items, save_frames,
                   select_frames_in_uniform_intervals)

DATA_PATH = 'data\\'
OUTPUT_DIR = 'shoppable_items\\'
MODEL = "yolov8s.pt"

if __name__ == "__main__":
    for video, ext in os.path.splitext(os.listdir(DATA_PATH)):
        selected_frames = select_frames_in_uniform_intervals(DATA_PATH, num_frames=20, blur_threshold=60, max_attempts=5)
        shoppable_items = detect_shoppable_items(selected_frames, MODEL)
        save_frames(shoppable_items, os.path.join(OUTPUT_DIR, video))
