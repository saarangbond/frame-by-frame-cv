import os
import json
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from metaflow import FlowSpec, step, Parameter
from bdd_util import (get_image_annotation_pairs, load_image_and_annotation, calculate_map)

class BDDFlow(FlowSpec):

    TRAINING_PATH = Parameter(
        'training-path', type=str,
        default="./bdd100k/images/100k/train/",
        help="The path to the BDD training images directory."
    )

    VAL_PATH = Parameter(
        'val-path', type=str,
        default="./bdd100k/images/100k/val/",
        help="The path to the BDD validation images directory."
    )

    EPOCHS = Parameter(
        'epochs', type=int,
        default=5,
        help = "Number of epochs for model training"
    )

    @step
    def start(self):
        self.next(self.load_data)

    @step
    def load_data(self):
        # Load the BDD100K data
        self.train_data = get_image_annotation_pairs(self.TRAINING_PATH)
        self.val_data = get_image_annotation_pairs(self.VAL_PATH)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.next(self.train_model)

    @step
    def train_model(self):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(self.EPOCHS):
            total_loss = 0
            for img_path, ann_path in self.train_data:
                image, target = load_image_and_annotation(img_path, ann_path)
                image = self.transforms(image)
                image = image.unsqueeze(0)
                
                optimizer.zero_grad()
                loss_dict = self.model(image, [target])
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()
                
                total_loss += losses.item()
            
            print(f"Epoch {epoch+1}/{self.EPOCHS}, Loss: {total_loss/len(self.train_data)}")
        
        self.next(self.eval_model)

    @step
    def eval_model(self):
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for img_path, ann_path in self.val_data:
                image, target = load_image_and_annotation(img_path, ann_path)
                image = self.transforms(image)
                image = image.unsqueeze(0)
                
                prediction = self.model(image)[0]
                
                all_preds.append(prediction)
                all_targets.append(target)
        
        # Calculate evaluation metrics (e.g., mAP)
        self.mAP = calculate_map(all_preds, all_targets)
        print(f"Mean Average Precision (mAP): {self.mAP}")
        
        self.next(self.end)

    @step
    def end(self):
        print("Object detection pipeline completed successfully.")


if __name__ == '__main__':
    BDDFlow()