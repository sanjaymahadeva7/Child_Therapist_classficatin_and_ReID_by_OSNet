from deep_sort_realtime.deepsort_tracker import DeepSort
import torchreid
import torch
import numpy as np
import cv2

class ReIDTracker:
    def __init__(self, weights_path):
        # Initialize the model
        self.reid_model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=False  # We will load weights manually
        )
        
        # Load pre-trained weights
        self.reid_model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        
        self.reid_model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reid_model.to(self.device)

        # Initialize DeepSort
        self.tracker = DeepSort(
            max_age=50,
            n_init=5,
            nms_max_overlap=0.7,
            max_cosine_distance=0.1
        )
        self.class_names = ['adult', 'child']

    def reid_embedder(self, image_crops):
        if len(image_crops) == 0:
            return np.empty((0, 1280)) 

        # Convert crops to tensors and send to device
        tensors = [self.preprocess(crop) for crop in image_crops]
        batch = torch.stack(tensors).to(self.device)
        
        # Extract embeddings
        with torch.no_grad():
            embeddings = self.reid_model(batch)
        return embeddings.cpu().numpy()

    def preprocess(self, crop):
        # Resize and normalize the image crops as required by OSNet
        input_image = cv2.resize(crop, (256, 128), interpolation=cv2.INTER_LINEAR)
        input_image = input_image.astype(np.float32) / 255.0
        input_image = (input_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        input_image = torch.tensor(input_image.transpose(2, 0, 1))
        return input_image


    def update(self, boxes, confidences, class_ids, frame):
        # Prepare detections for DeepSort
        detections = []
        for i, box in enumerate(boxes):
            if confidences[i] > 0.7:
                # Add a placeholder for embeddings (if needed)
                detections.append((box, confidences[i], self.class_names[int(class_ids[i])]))

        # Update tracker with new detections
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        return tracks
