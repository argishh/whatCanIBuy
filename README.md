# **What Can I Buy?**

Identify **shoppable items** from videos. It ensures computational efficiency by:
1. Dividing the video into uniform intervals for sampling frames within each interval.
2. Selecting non-blurry frames using a blur detection algorithm.
3. Annotating the selected frames with bounding boxes and labels using a **YOLOv8 object detection model**.

It automatically selects **20 diverse frames** from the video and performs shoppable item detection on the selected frames. This is done to maintain a balance between computational efficiency and the quantity frames processed.

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/argishh/whatCanIBuy.git
   cd whatCanIBuy
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

(Optional) : Downloading videos from YouTube

3. Download [`yt-dlp.exe`](https://github.com/yt-dlp/yt-dlp?tab=readme-ov-file#recommended) and store it in `C:\yt-dlp\` (for windows)
5. Add `C:\yt-dlp\` to `PATH` (for windows)
6. Ensure `ffmpeg` is installed

## **Usage**

1. Place input videos in the `data/` directory.
2. Run the main script:
   ```bash
   python main.py
   ```

## **Dependencies**
- **Python 3.8+**
- **OpenCV**: For frame extraction and blur detection.
- **Ultralytics YOLOv8**: For object detection and annotation.
- **NumPy**: For numerical operations.

## **Configuration**

You can customize the following parameters in the script:

- **Blur Threshold (`blur_threshold`)**:  
  Controls the sensitivity of the blur detection. Higher values are stricter.  
  Default: `80`.

- **Number of Frames (`num_frames`)**:  
  Number of frames to select from the video.  
  Default: `20`.

- **Max Sampling Attempts (`max_attempts`)**:  
  Maximum number of attempts to find a sharp frame within an interval.  
  Default: `5`.

- **YOLO Model (`yolov8n.pt`)**:  
  The YOLOv8 model is used for object detection. You can select any other yolo model from [docs.ultralytics.com](https://docs.ultralytics.com/models/yolov8/#performance-metrics)


## **Future Improvements**
- Integrate additional post-processing for detected objects.
- Implementing more frame selection methods (Other than uniform intervals).
- Enhance blur detection for edge cases (e.g., motion blur).

