import os

import cv2
from rich import print
from tqdm import tqdm
from ultralytics import YOLO


def is_frame_blurry(frame, threshold=80):
    """
    Determine if a frame is blurry using laplacian variance.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold


def select_frames_in_uniform_intervals(video_path, num_frames=20, blur_threshold=80, max_attempts=5):
    """
    Select non-blurry frames by dividing the video into intervals and sampling frames within each interval.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_size = total_frames // num_frames  # Divide the video into equal intervals
    
    selected_frames = []

    for i in tqdm(range(num_frames), desc="Parsing Video", colour="green"):
        first_frame = i * interval_size
        sharp_frame = None
        sharp_not_found_idx = []
        
        for attempt in range(max_attempts):
            # Randomly sample a frame within the interval
            frame_idx = first_frame + attempt//interval_size 
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            # Check if the frame is read correctly
            if not ret:
                print(f"\n[red1][WARNING] Could not read frame at index {frame_idx}.[/]")
                continue

            # Check if the frame is blurry
            if not is_frame_blurry(frame, blur_threshold):
                sharp_frame = frame
                break

        # If no sharp frame is found after max_attempts, select the last attempted frame
        if sharp_frame is not None:
            selected_frames.append(sharp_frame)
        else:
            sharp_not_found_idx.append(str(i+1))
            selected_frames.append(frame)

    if sharp_not_found_idx != []:
        print(f"\n[red1][WARNING] No sharp frame found in intervals [{", ".join(sharp_not_found_idx)}]. Using last sampled frame.[/]")

    cap.release()
    return selected_frames


def detect_shoppable_items(frames, model="yolov8n.pt"):
    """
    Add bounding boxes with labels to the frames using YOLOv8.
    """
    model = YOLO(model)
    items = []

    print('\n[green1]Detecting shoppable items...[/]')
    for frame in frames:
        results = model(frame)
        annotated_frame = frame.copy()

        for result in results:
            for box in result.boxes.data.tolist():
                x1, y1, x2, y2, confidence, cls = box
                label = f"{result.names[int(cls)]} {confidence:.2f}"

                # Draw bounding box
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw label
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                label_y = max(int(y1) - 10, label_size[1])
                cv2.rectangle(
                    annotated_frame,
                    (int(x1), label_y - label_size[1]),
                    (int(x1) + label_size[0], label_y),
                    (0, 255, 0),
                    -1,)
                
                cv2.putText(
                    annotated_frame,
                    label,
                    (int(x1), label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,)

        items.append(annotated_frame)
    return items


def save_frames(frames, output_dir):
    """
    Save selected frames to a directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, frame in enumerate(frames):
        frame_path = os.path.join(output_dir, f"frame_{idx+1:02d}.jpg")
        cv2.imwrite(frame_path, frame)
    print(f"\n[green1][INFO] Saved frames to {output_dir}[/]")
