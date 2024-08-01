from metaflow import FlowSpec, step, Parameter, kubernetes, card, current, pypi_base
from metaflow.cards import Markdown, Image, Artifact, ProgressBar, VegaChart

@pypi_base(python='3.11.9', packages={
    'torch': '2.4.0',
    'torchvision': '0.19.0',
    'pandas': '2.2.0',
    'numpy': '1.24.3',
    'scikit-learn': '1.5.1',
    'matplotlib': '3.9.1',
    'opencv-python-headless': '4.10.0.84',
    'tqdm': '4.66.4',
    'pillow': '10.4.0',
    'mapcalc': '0.2.2',
    'altair': '5.2.0'
})
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

    NUM_WORKERS = Parameter(
        'num-workers', type=int,
        default=0,
        help="Number of workers to use for dataset loading tasks."
    )

    SPLITS = Parameter(
        'splits', type=int,
        default = None,
        help="Number of parallel image loading tasks to be run."
    )

    IMG_LOADING_BATCH_SIZE = Parameter(
        'loading-batch-size', type=int,
        default=None,
        help="Number of images to be loaded in a singular batch. Superseded by splits."
    )

    CAP = Parameter(
        'cap', type=int,
        default=None,
        help="Cap the number of training images to run this flow on"
    )

    PRETRAINED = Parameter(
        'pretrained', type=bool,
        default=True,
        help="Whether to use a pretrained cv model or not."
    )

    @step
    def start(self):
        self.next(self.load_annotations)

    @step
    def load_annotations(self):
        import os
        import json
        import numpy as np

        # Load the JSON annotations
        with open(os.path.join(self.ANN_PATH, 'det_train.json'), 'r') as f:
            self.train_annotations = json.load(f)
        with open(os.path.join(self.ANN_PATH, 'det_val.json'), 'r') as f:
            self.val_annotations = json.load(f)

        if(self.CAP):
            self.train_annotations = self.train_annotations[:self.CAP]
            self.val_annotations = self.val_annotations[:(self.CAP // 2)]

        print(f"Loaded {len(self.train_annotations)} training annotations.")
        print(f"Loaded {len(self.val_annotations)} validation annotations.")

        num_train_splits = 1
        num_val_splits = 1

        if self.SPLITS:
            num_train_splits = self.SPLITS
            num_val_splits = self.SPLITS
        elif self.IMG_LOADING_BATCH_SIZE:
            num_train_splits, r1 = divmod(len(self.train_annotations), self.IMG_LOADING_BATCH_SIZE)
            num_train_splits += bool(r1)
            num_val_splits, r2 = divmod(len(self.val_annotations), self.IMG_LOADING_BATCH_SIZE)
            num_val_splits += bool(r2)

        list_train_annotations = np.array_split(self.train_annotations, num_train_splits)
        list_val_annotations = np.array_split(self.val_annotations, num_val_splits)

        self.split_train_annotations = []
        for ann in list_train_annotations:
            self.split_train_annotations.append({
                'type': 'train',
                'annotations': ann,
            })

        self.split_val_annotations = []
        for ann in list_val_annotations:
            self.split_val_annotations.append({
                'type': 'val',
                'annotations': ann,
            })

        self.all_split_annotations = [self.split_train_annotations, self.split_val_annotations]

        self.next(self.launch_images, foreach='all_split_annotations')
    
    @step
    def launch_images(self):
        self.split_annotations = self.input
        self.next(self.load_images, foreach='split_annotations')

    @step
    def load_images(self):
        import os
        from bdd_util import process_annotations

        ann_object = self.input
        ann_type = ann_object['type']
        annotations = ann_object['annotations']
        self.small_ann_images = []
        try:
            # Load and preprocess training images
            self.small_ann_images = process_annotations(annotations, os.path.join(self.IMAGES_PATH, ann_type), ann_type)
        except Exception as e:
            print(f'Error during image preprocessing: {e}')
            raise
        self.next(self.join_load_images)

    @step
    def join_load_images(self, inputs):
        self.ann_images = []
        for inp in inputs:
            self.ann_images.extend(inp.small_ann_images)
        self.next(self.compile_images)

    @step
    def compile_images(self, inputs):
        self.all_ann_images = [inp.ann_images for inp in inputs]
        self.train_images = []
        self.val_images = []
        self.train_annotations = []
        self.val_annotations = []

        for ann_list in self.all_ann_images:
            for ann_obj in ann_list:
                if(ann_obj['type'] == 'train'):
                    self.train_images.append(ann_obj['image'])
                    self.train_annotations.append(ann_obj['annotation'])
                elif(ann_obj['type'] == 'val'):
                    self.val_images.append(ann_obj['image'])
                    self.val_annotations.append(ann_obj['annotation'])
        self.next(self.train_model)

    @card(type='blank', refresh_interval=0.1)
    #@kubernetes(memory=16000, shared_memory=1024)
    @kubernetes(memory=16000, shared_memory=1024, gpu=1)
    @step
    def train_model(self):
        import torch
        from torch.utils.data import DataLoader
        import torchvision.transforms as T
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        import altair as alt
        import pandas as pd
        from bdd_util import BDD100KDataset

        # Create datasets and dataloaders
        train_dataset = BDD100KDataset(self.train_images, self.train_annotations, transforms=T.ToTensor())
        val_dataset = BDD100KDataset(self.val_images, self.val_annotations, transforms=T.ToTensor())
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=self.NUM_WORKERS, collate_fn=lambda x: tuple(zip(*x)))

        # Load a pretrained model and modify it for our dataset
        model = fasterrcnn_resnet50_fpn(pretrained=self.PRETRAINED)
        #num_classes = len(set([label['category'] for annotation in self.train_annotations for label in annotation['labels']])) + 1  # Assuming labels start from 1
        num_classes = 10 + 1
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        # Define optimizer and learning rate
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        self.epoch_metrics = pd.DataFrame({'epoch': [], 'loss': []})
        alt_chart = (
            alt.Chart(self.epoch_metrics, title=alt.TitleParams('Loss vs Epoch', anchor='middle'))
            .mark_line(color='red', point=True)
            .encode(x="epoch", y="loss")
            .interactive()
        )
        chart = VegaChart.from_altair_chart(alt_chart)
        current.card.append(chart)
        current.card.refresh()

        progressbar = ProgressBar(max=(self.EPOCHS*len(train_dataset)), label='Training Progress')
        current.card.append(progressbar)
        current.card.refresh()

        # Training loop
        num_epochs = self.EPOCHS
        for epoch in range(num_epochs):
            model.train()
            i = 0
            epoch_loss = 0
            for images, targets in train_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                epoch_loss += losses.item()
                if i % 100 == 0:
                    print(f"Epoch: {epoch}, Iteration: {i}, Loss: {losses.item()}")
                i += 1
                progressbar.update(epoch*len(train_dataset)+i)
                current.card.refresh()

            lr_scheduler.step()
            epoch_loss /= i
            new_row = pd.DataFrame({'epoch': [epoch], 'loss': [epoch_loss]})
            self.epoch_metrics = pd.concat([self.epoch_metrics, new_row], ignore_index=True)
            alt_chart = (
                alt.Chart(self.epoch_metrics, title=alt.TitleParams('Loss vs Epoch', anchor='middle'))
                .mark_line(color='red', point=True)
                .encode(x="epoch", y="loss")
                .interactive()
            )
            chart.update(alt_chart.to_dict())
            current.card.refresh()
            print(f"Epoch {epoch} finished.")

        progressbar.update(self.EPOCHS*len(train_dataset))
        self.model = model
        self.next(self.evaluate_model)

    #@kubernetes(memory=16000, shared_memory=1024)
    @kubernetes(memory=16000, shared_memory=1024, gpu=1)
    @step
    def evaluate_model(self):
        import torch
        from torch.utils.data import DataLoader
        import torchvision.transforms as T
        import numpy as np
        from mapcalc import calculate_map
        import cv2
        from bdd_util import BDD100KDataset

        model = self.model
        model.eval()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        val_dataset = BDD100KDataset(self.val_images, self.val_annotations, transforms=T.ToTensor())
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=self.NUM_WORKERS, collate_fn=lambda x: tuple(zip(*x)))

        mAP_scores = []
        self.sample_images = []

        # Evaluation loop
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                outputs = model(images)
                
                for i, output in enumerate(outputs):
                    pred_object = {
                        'boxes': output['boxes'].cpu().numpy(),
                        'labels': output['labels'].cpu().numpy(),
                        'scores': output['scores'].cpu().numpy()
                    }

                for target in targets:
                    target_object = {
                        'boxes': target['boxes'].cpu().numpy(),
                        'labels': target['labels'].cpu().numpy()
                    }

                mAP_scores.append(calculate_map(pred_object, target_object, 0.5))
                # Save sample images
                if len(self.sample_images) < 9:
                    image = images[i].cpu().numpy().transpose(1, 2, 0)
                    image = (image * 255).astype(np.uint8)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    for box in pred_object['boxes']:
                        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                    for box in target_object['boxes']:
                        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    self.sample_images.append(image)
                
        self.mAP = np.mean(mAP_scores)

        print("Evaluation complete.")
        self.next(self.render_card)

    @card
    @step
    def render_card(self):
        import io
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image as Img

        #Populate card content
        current.card.append(Markdown('# Training Metrics'))
        current.card.append(Markdown('Mean Average Precision (mAP):'))
        current.card.append(Artifact(self.mAP))

        num_images = len(self.sample_images)
        grid_size = int(np.ceil(np.sqrt(num_images)))

        # Render grid of sample images
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(20,20))
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            if(i < num_images):
                ax.imshow(self.sample_images[i])
            ax.axis('off')

        grid_buf = io.BytesIO()
        plt.savefig(grid_buf, format='png', dpi=100)
        grid_img = Img.open(grid_buf).convert('RGB')

        current.card.append(Markdown('# Sample Predictions'))
        current.card.append(Markdown('The predictions are displayed in red while the ground truth is displayed in green.'))
        current.card.append(Image.from_pil_image(grid_img))

        grid_buf.close()

        self.next(self.end)

    @step
    def end(self):
        print(f'Mean Average Precision (mAP): {self.mAP}')
        print("Object detection pipeline completed successfully.")

if __name__ == '__main__':
    BDDFlow()