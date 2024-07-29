import torch
from torch.utils.data import Dataset
import cv2
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

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
        target = {
            'boxes': torch.as_tensor([obj['bbox'] for obj in annotation['labels']], dtype=torch.float32),
            'labels': torch.as_tensor([obj['category_id'] for obj in annotation['labels']], dtype=torch.int64)
        }
        if self.transforms:
            image = self.transforms(image)
        return image, target


def calculate_map(predictions, targets):
        """
        Calculate Mean Average Precision (mAP) for the predictions using pycocotools.
        """
        coco_gt, coco_dt = convert_to_coco_format(predictions, targets)
        
        with open("temp_gt.json", "w") as f:
            json.dump(coco_gt, f)
        
        with open("temp_dt.json", "w") as f:
            json.dump(coco_dt, f)
        
        coco_gt = COCO("temp_gt.json")
        coco_dt = coco_gt.loadRes("temp_dt.json")
        
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        return coco_eval.stats[0]  # Return mAP

def convert_to_coco_format(predictions, targets):
    """
    Convert predictions and targets to COCO format.
    """
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": str(i)} for i in range(1, 21)]  # Adjust categories as needed
    }
    
    coco_dt = []
    
    ann_id = 1
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        img_info = {
            "id": i,
            "width": target["boxes"].size(1),
            "height": target["boxes"].size(0)
        }
        coco_gt["images"].append(img_info)
        
        for j, (box, label) in enumerate(zip(target["boxes"], target["labels"])):
            ann = {
                "id": ann_id,
                "image_id": i,
                "category_id": label.item(),
                "bbox": [box[0].item(), box[1].item(), box[2].item() - box[0].item(), box[3].item() - box[1].item()],
                "area": (box[2] - box[0]).item() * (box[3] - box[1]).item(),
                "iscrowd": 0
            }
            coco_gt["annotations"].append(ann)
            ann_id += 1
        
        for j, (box, score) in enumerate(zip(pred["boxes"], pred["scores"])):
            ann = {
                "image_id": i,
                "category_id": pred["labels"][j].item(),
                "bbox": [box[0].item(), box[1].item(), box[2].item() - box[0].item(), box[3].item() - box[1].item()],
                "score": score.item()
            }
            coco_dt.append(ann)
    
    return coco_gt, coco_dt
