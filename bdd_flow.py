import os
import json
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from metaflow import FlowSpec, step, Parameter
from bdd_util import (BDD100KDataset, calculate_map)

class BDDFlow(FlowSpec):

    IMAGES_PATH = Parameter(
        'images-path', type=str,
        default='./bdd100k/images/',
        help='Path to image directory.'
    )

    ANN_PATH = Parameter(
        'ann-path', type=str,
        default='./bdd100k/annotations/',
        help='Path to annotations directory.'
    )

    EPOCHS = Parameter(
        'epochs', type=int,
        default=5,
        help = "Number of epochs for model training."
    )

    @step
    def start(self):
        self.next(self.load_annotations)

    @step
    def load_annotations(self):
        # Load the JSON annotations
        with open(os.path.join(self.ANN_PATH, 'det_train.json'), 'r') as f:
            self.train_annotations = json.load(f)
        with open(os.path.join(self.ANN_PATH, 'det_val.json'), 'r') as f:
            self.val_annotations = json.load(f)

        print(f"Loaded {len(self.train_annotations)} training annotations.")
        print(f"Loaded {len(self.val_annotations)} validation annotations.")
        self.next(self.preprocess_images)

    @step
    def preprocess_images(self):
        self.train_images = []
        self.val_images = []

        # Load and preprocess training images
        for annotation in self.train_annotations:
            image_path = annotation['name']
            image_id = image_path.split('.')[0]
            image = Image.open(os.path.join(self.IMAGES_PATH, 'train', image_path)).convert('RGB')
            image = np.array(image)
            self.train_images.append((image, annotation))

        # Load and preprocess validation images
        for annotation in self.val_annotations:
            image_path = annotation['name']
            image_id = image_path.split('.')[0]
            image = Image.open(os.path.join(self.IMAGES_PATH, 'val', image_path)).convert('RGB')
            image = np.array(image)
            self.val_images.append((image, annotation))

        print(f"Preprocessed {len(self.train_images)} training images.")
        print(f"Preprocessed {len(self.val_images)} validation images.")
        self.next(self.train_model)

    @step
    def train_model(self):
        # Create datasets and dataloaders
        train_dataset = BDD100KDataset(self.train_images, self.train_targets, transforms=T.ToTensor())
        val_dataset = BDD100KDataset(self.val_images, self.val_targets, transforms=T.ToTensor())
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

        # Load a pretrained model and modify it for our dataset
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = len(set([label['category_id'] for annotation in self.train_targets for label in annotation['labels']])) + 1  # Assuming labels start from 1
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        # Define optimizer and learning rate
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        # Training loop
        num_epochs = self.num_epochs
        for epoch in range(num_epochs):
            model.train()
            i = 0
            for images, targets in train_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                if i % 100 == 0:
                    print(f"Epoch: {epoch}, Iteration: {i}, Loss: {losses.item()}")
                i += 1

            lr_scheduler.step()
            print(f"Epoch {epoch} finished")

        self.model = model
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        model = self.model
        model.eval()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        val_dataset = BDD100KDataset(self.val_images, self.val_targets, transforms=T.ToTensor())
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

        all_predictions = []
        all_targets = []

        # Evaluation loop
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                outputs = model(images)
                all_predictions.extend(outputs)
                all_targets.extend(targets)

        self.mAP = calculate_map(all_predictions, all_targets)

        print("Evaluation complete.")
        self.next(self.end)

    @step
    def end(self):
        print(f'Mean Average Precision (mAP): {self.mAP}')
        print("Object detection pipeline completed successfully.")


if __name__ == '__main__':
    BDDFlow()