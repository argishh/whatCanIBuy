import cv2
from ultralytics import YOLO
from rich import print
from select_frames import select_frames_in_uniform_intervals, save_frames

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

if __name__ == "__main__":
    VIDEO_PATH = 'data\\video.mp4'
    OUTPUT_DIR = 'shoppable_items'

    # Load YOLOv8 model
    model = "yolov8s.pt"

    selected_frames = select_frames_in_uniform_intervals(VIDEO_PATH, num_frames=20, blur_threshold=60, max_attempts=5)
    shoppable_items = detect_shoppable_items(selected_frames, model)

    # Saving the shoppable items' extracted frames
    save_frames(shoppable_items, OUTPUT_DIR)