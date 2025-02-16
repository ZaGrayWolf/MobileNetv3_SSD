import os
import torch
import torchvision
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
import cv2
import torch.optim as optim
import torch.utils.data as data
from torchvision.ops.boxes import box_iou
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

# Define image preprocessing transforms with Augmentations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((320, 320)),  # Resize to model's expected input size
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class VOCDataset(data.Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.annotations = []
        self.class_names = set()

        for annotation_file in os.listdir(dataset_path):
            if annotation_file.endswith('.xml'):
                annotation = self._parse_annotation(os.path.join(dataset_path, annotation_file))
                image_path = os.path.join(dataset_path, annotation['filename'])
                if os.path.exists(image_path):
                    annotation['image_path'] = image_path
                    self.annotations.append(annotation)
                    self.class_names.update(obj['name'] for obj in annotation['objects'])

        self.class_names = sorted(list(self.class_names))
        self.class_dict = {name: i + 1 for i, name in enumerate(self.class_names)}

    def _parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        return {
            'filename': root.find('filename').text,
            'objects': [
                {
                    'name': obj.find('name').text,
                    'bbox': [
                        int(obj.find('bndbox/xmin').text),
                        int(obj.find('bndbox/ymin').text),
                        int(obj.find('bndbox/xmax').text),
                        int(obj.find('bndbox/ymax').text)
                    ]
                } for obj in root.findall('object')
            ]
        }

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        image = cv2.imread(annotation['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        image = transform(image)

        # Handle case with no objects
        if not annotation['objects']:
            return image, {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64)
            }

        boxes = torch.tensor([obj['bbox'] for obj in annotation['objects']], dtype=torch.float32)
        labels = torch.tensor([self.class_dict[obj['name']] for obj in annotation['objects']], dtype=torch.int64)

        return image, {'boxes': boxes, 'labels': labels}

# Configuration
TRAIN_DATASET_PATH = "/kaggle/input/rccup-voc2/ROBOCUP_OBJECTS_2024.v1-yolov3_jetson.voc/train"
VAL_DATASET_PATH = "/kaggle/input/rccup-voc2/ROBOCUP_OBJECTS_2024.v1-yolov3_jetson.voc/val"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Training and Validation Datasets
train_dataset = VOCDataset(TRAIN_DATASET_PATH)
val_dataset = VOCDataset(VAL_DATASET_PATH)

train_loader = data.DataLoader(
    train_dataset, 
    batch_size=4, 
    shuffle=True, 
    drop_last=True,
    collate_fn=lambda x: tuple(zip(*x))
)

val_loader = data.DataLoader(
    val_dataset, 
    batch_size=4, 
    shuffle=False,
    collate_fn=lambda x: tuple(zip(*x))
)

# Model Setup
num_classes = 23  # Adding 1 for background class
model = ssdlite320_mobilenet_v3_large(num_classes=num_classes).to(device)

# Optimizer and Scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR every 10 epochs

# Focal Loss
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

loss_fn = FocalLoss()

# Function to compute Validation IoU
def evaluate_model(model, dataloader, device):
    model.eval()
    total_iou = 0
    total_images = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            
            for output, target in zip(outputs, targets):
                if len(target["boxes"]) == 0 or len(output["boxes"]) == 0:
                    continue  # Skip images with no objects
                
                # Normalize boxes to the same scale
                pred_boxes = output["boxes"] / 320.0
                true_boxes = target["boxes"] / 320.0

                iou = box_iou(pred_boxes, true_boxes)
                max_iou, _ = iou.max(dim=1)  # Get max IoU per predicted box
                
                total_iou += max_iou.mean().item()  # Mean IoU per image
                total_images += 1

    return total_iou / total_images if total_images > 0 else 0

# Training Loop 
for epoch in range(50):
    model.train()
    total_loss = 0.0

    for images, targets in train_loader:
        images = torch.stack(images).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    # Validation Accuracy
    val_accuracy = evaluate_model(model, val_loader, device)

    print(f"Epoch [{epoch+1}/50], Loss: {total_loss / len(train_loader):.4f}, Validation IoU: {val_accuracy:.4f}")

    # Save model after each epoch
    torch.save(model.state_dict(), f"ssdlite_mobilenet_v3_large_voc_epoch{epoch+1}.pth")

    # Step the scheduler
    scheduler.step()
