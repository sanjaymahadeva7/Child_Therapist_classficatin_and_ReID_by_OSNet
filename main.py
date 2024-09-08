import cv2
import time
from model_loader import load_model
from retracker import ReIDTracker

# Path to the pre-trained model weights
weights_path = 'osnet_x1_0_imagenet.pth'

# Initialize the ReIDTracker with weights path
tracker = ReIDTracker(weights_path)

# Load the YOLO model
model = load_model('best.pt')

# Load the video file
input_video_path = r'C:\Users\My PC\OneDrive\Desktop\children and therapist project\test_videos\v13.mp4'
cap = cv2.VideoCapture(input_video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get original FPS and frame size of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define playback speed multiplier
playback_speed = 0.85

# Calculate the frame skip interval and delay
frame_skip_interval = int(playback_speed)
frame_time = 1 / fps
frame_delay = int(frame_time * 1000 / playback_speed)

#VideoWriter object to save the output video
output_video_path = r'C:\Users\My PC\OneDrive\Desktop\children and therapist project\test_videos\output_v13.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps * playback_speed, (frame_width, frame_height))

def process_frame(frame):
    # Run YOLO inference
    results = model(frame)

    # Extract bounding boxes, confidences, and labels
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    # Update tracker with detections
    tracks = tracker.update(boxes, confidences, class_ids, frame)

    # Draw bounding boxes and track IDs
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        # Extract bounding box coordinates and ID
        box = track.to_tlbr() 
        track_id = track.track_id
        class_name = track.get_det_class() 

        # Draw bounding box and label with track ID
        color = (0, 255, 0) if class_name == "adult" else (255, 0, 0)
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(frame, f'ID: {track_id} {class_name}', (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return frame

frame_count = 0
while cap.isOpened():
    for _ in range(frame_skip_interval):
        ret, _ = cap.read()
        if not ret:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            exit()

    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    
    # Process the frame
    frame = process_frame(frame)
    
    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow('YOLO Tracking', frame)

    # Calculate delay
    elapsed_time = time.time() - start_time
    delay = max(1, int(frame_delay - elapsed_time * 1000))
    
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
