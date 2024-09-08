# ReID and YOLOv8 Tracking with Video Processing

## Overview

This project integrates YOLOv8 for object detection and OSNet for re-identification (ReID) of adults and children in a video. The goal is to detect objects in real-time, track them using their ReID embeddings, and improve the tracking consistency even after occlusion or re-entry into the frame.

The process involves loading a video file, running YOLOv8 for detection, using DeepSORT for object tracking, and applying OSNet for person re-identification. The video is processed at a custom playback speed, with the option to save the processed video.

## Project Components

### 1. YOLOv8 Detection

YOLOv8 is used to detect objects such as adults and children in each frame of the video.  
The `load_model` function loads the trained YOLOv8 model, and inference is run on each frame to generate bounding boxes, object classes, and confidence scores.  
These detections are passed to the `ReIDTracker`, which performs the next steps of tracking and embedding.

### 2. OSNet ReID Model

We use a pre-trained OSNet model for person re-identification. This allows the model to assign unique IDs to each person in the video and keep track of them across frames, even if they leave and re-enter the scene.  
The OSNet model processes cropped detections and generates embeddings (high-dimensional vectors) that serve as unique signatures for each detected person.  
The tracker compares these embeddings across frames to maintain consistent ID assignment.

### 3. DeepSORT Tracking

DeepSORT is a popular multi-object tracking algorithm used here to associate object detections with tracked objects.  
The bounding boxes, confidence scores, and class IDs from YOLOv8 are passed to DeepSORT, along with the ReID embeddings from OSNet, to track people across frames.  
The tracker's goal is to maintain consistent tracking even during occlusions or re-entries.

### 4. Video Processing Pipeline

The video is read frame by frame using OpenCV, and detection and tracking are applied to each frame.  
The processed frames are displayed in real-time, with bounding boxes drawn around detected objects and labels showing the track ID and class (e.g., adult or child).  
The playback speed can be adjusted using a multiplier.  
The processed video can also be saved as an output file.

## Data Preparation for YOLO Model

For the YOLO model training, I manually collected a dataset consisting of over 1000 images. These images were specifically selected to capture different scenarios relevant to the project requirements. To ensure high-quality annotations, I utilized **Roboflow** for the data preparation process.

### The steps included:

- **Manual Data Collection:** I gathered a diverse set of images that represented various angles, lighting conditions, and environments to enhance model generalization.
  
- **Annotation:** Using Roboflow’s annotation tools, I labeled each object of interest in the images. This involved defining bounding boxes for the key classes (e.g., adults, children) across all images, ensuring that the annotations were precise and consistent.
  
- **Data Augmentation:** To further improve the robustness of the model, I applied a range of augmentations within Roboflow, such as random flips, rotations, and color variations. These augmentations helped increase the variety of training data, enabling the model to handle diverse real-world conditions.
  
- **Dataset Splitting:** The dataset was then split into training, validation, and test sets, ensuring that the model could be trained effectively while also having unseen data for evaluation.

After preparing the dataset, I used it to fine-tune the YOLO model to suit the specific objectives of the project, such as detecting and tracking different individuals in the video. The accuracy and efficiency of the annotations were critical to achieving high performance in object detection and tracking.

## How to Run the Project

### 1. Prerequisites

- Python 3.8+
- Install the necessary dependencies using `requirements.txt`:
  
```bash
pip install -r requirements.txt
```

### 2. YOLOv8 Model

The YOLOv8 model is pre-trained and saved as `best.pt`. This model detects adults and children in video frames. Ensure the model is placed in the same directory or update the path in the code accordingly.

### 3. Pre-trained OSNet Model

The ReID model is pre-trained and saved as `osnet_x1_0_imagenet.pth`. Place this file in the working directory or update the path in the code as needed.

### 4. Running the Program

To run the video processing, execute the `main.py` script:

```bash
python main.py
```

Ensure that the path to the video file is correct. This script will load the YOLOv8 model, the ReID model, and the video, and then start processing.

### 5. Output Video

The processed video, with bounding boxes and tracking IDs, will be saved in the project directory as `output_video.avi`. You can adjust the filename in the script. And for better experience i have converted all the output videos to mp4 formate. All output videos are stored in Output_videos

## Analyzing the Model Predictions

### Detection:

- The YOLOv8 model outputs bounding boxes, confidence scores, and class IDs for each detected object.
- Objects are classified as either "adult" or "child" based on the trained model.
- Predictions are displayed on the video as bounding boxes with labels showing the class and confidence score.

### Tracking:

- The DeepSORT tracker assigns a unique track ID to each detected person in the frame.
- The OSNet model generates embeddings for detected persons, which are used to match them across frames. If a person leaves the frame and re-enters later, the same ID is reassigned based on the embedding.

### Embedding and Re-identification:

- OSNet generates high-dimensional vectors (embeddings) for each detected person. These embeddings help to maintain consistent ID assignment even after occlusion or re-entry.
- The embeddings are compared across frames using cosine similarity to determine whether two detections represent the same person.

### Speed and Performance:

- The code is optimized to speed up video playback. The frame rate can be adjusted by modifying the `playback_speed` multiplier.
- The `cv2.VideoWriter` is used to save the processed video with all detection and tracking information.

## Troubleshooting

### 1. Unsupported Video Formats

If you encounter unsupported video formats, ensure that the video codec is compatible with OpenCV. Try converting the video to `.mp4` or `.avi` formats using a tool like **FFmpeg**.

### 2. Slow Video Processing

Ensure that you are running the script on a machine with a GPU for faster inference. Adjust the playback speed to a higher value to improve real-time performance.
