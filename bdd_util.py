import torch
from torch.utils.data import Dataset
import torchvision.models.detection as detection
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sklearn
from sklearn.metrics import average_precision_score

class BDD100KDataset(Dataset):
    def __init__(self, images, annotations, transforms=None):
        self.images = images
        self.annotations = annotations
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        annotation = self.annotations[idx]
        
        boxes = []
        labels = []

        category_mapping = {
            'pedestrian': 1,
            'car': 2,
            'truck': 3,
            'bus': 4,
            'traffic light': 5,
            'traffic sign': 6,
            'rider': 7,
            'train': 8,
            'motorcycle': 9,
            'bicycle': 10,
        }

        for obj in annotation['labels']:
            box = obj['box2d']
            boxes.append([box['x1'], box['y1'], box['x2'], box['y2']])
            labels.append(category_mapping[obj['category']])
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, target

def process_annotations(annotations, image_dir, ann_type):
    processed_images = []
    for annotation in tqdm(annotations, desc='Processing Images'):
        image, image_path = process_image(annotation['name'], image_dir)
        if image is not None:
            processed_images.append({
                'image': image,
                'type': ann_type,
                'annotation': annotation
            })
        else:
            print(f"Failed to process {image_path}")
    return processed_images

def process_image(image_path, image_dir):
    try:
        image = Image.open(os.path.join(image_dir, image_path)).convert('RGB')
        return np.array(image), image_path  # Return the image and its path for debugging
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, image_path

def calculate_map(predictions, targets, iou_threshold=0.5):
    # Assuming predictions and targets are lists of dictionaries
    # Each dictionary should have keys: 'boxes' and 'labels'

    all_true_labels = []
    all_pred_labels = []
    all_scores = []

    for pred, target in zip(predictions, targets):
        true_boxes = target['boxes'].cpu().numpy()
        true_labels = target['labels'].cpu().numpy()
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()

        all_true_labels.append(true_labels)
        all_pred_labels.append(pred_labels)
        all_scores.append(scores)

    # Flatten lists
    all_true_labels = np.concatenate(all_true_labels)
    all_pred_labels = np.concatenate(all_pred_labels)
    all_scores = np.concatenate(all_scores)

    # Calculate mAP
    mAP = average_precision_score(all_true_labels, all_scores)
    return mAP
