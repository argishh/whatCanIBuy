import cv2
import os
from tqdm import tqdm
from rich import print

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
