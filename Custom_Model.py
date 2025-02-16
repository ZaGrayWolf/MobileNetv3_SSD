import os
import torch
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
import cv2
import torch.optim as optim
import torch.utils.data as data
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import torch.nn.functional as F
from torch import nn
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.models.detection.image_list import ImageList
from collections import OrderedDict
from torchvision.ops import batched_nms  # Import batched_nms directly

# ---------------------------------------------------------------------------
# Transforms for training and validation
train_transform = transforms.Compose([
    transforms.ToPILImage(),      # Convert NumPy array (from cv2) to PIL Image
    transforms.Resize((320, 320)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),        # Convert PIL Image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((320, 320)),
    transforms.ToTensor(),        # Convert PIL Image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------------------------------------
# VOCDataset: Returns a tensor and a target dictionary
class VOCDataset(data.Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.annotations = []
        self.class_names = set()
        for annotation_file in os.listdir(dataset_path):
            if annotation_file.endswith('.xml'):
                try:
                    annotation = self._parse_annotation(os.path.join(dataset_path, annotation_file))
                    image_path = os.path.join(dataset_path, annotation['filename'])
                    if os.path.exists(image_path):
                        annotation['image_path'] = image_path
                        self.annotations.append(annotation)
                        self.class_names.update(obj['name'] for obj in annotation['objects'])
                except Exception as e:
                    print(f"Error parsing {annotation_file}: {e}")
                    continue
        self.class_names = sorted(list(self.class_names))
        self.class_dict = {name: i + 1 for i, name in enumerate(self.class_names)}

    def _parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        filename = root.find('filename')
        if filename is None:
            filename = os.path.basename(annotation_path).replace('.xml', '.jpg')
        else:
            filename = filename.text
        return {
            'filename': filename,
            'objects': [
                {
                    'name': obj.find('name').text,
                    'bbox': [
                        float(obj.find('bndbox/xmin').text),
                        float(obj.find('bndbox/ymin').text),
                        float(obj.find('bndbox/xmax').text),
                        float(obj.find('bndbox/ymax').text)
                    ]
                } for obj in root.findall('object')
            ]
        }

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        try:
            image_np = cv2.imread(annotation['image_path'])
            if image_np is None:
                raise ValueError(f"Failed to load image: {annotation['image_path']}")
            # cv2 returns BGR; convert to RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            # Apply transform (should return a tensor)
            if self.transform:
                image = self.transform(image_np)
            else:
                image = transforms.ToTensor()(Image.fromarray(image_np))
            
            if not annotation['objects']:
                print(f"Index {index}: No objects found, returning image of type {type(image)}")
                return image, {
                    'boxes': torch.zeros((0, 4), dtype=torch.float32),
                    'labels': torch.zeros(0, dtype=torch.int64)
                }
            
            boxes = torch.tensor([obj['bbox'] for obj in annotation['objects']], dtype=torch.float32)
            labels = torch.tensor([self.class_dict[obj['name']] for obj in annotation['objects']], dtype=torch.int64)
            print(f"Index {index}: Returning image of type {type(image)}")
            return image, {'boxes': boxes, 'labels': labels}
        except Exception as e:
            print(f"Error loading item {index}: {e}")
            dummy = torch.zeros((3, 320, 320), dtype=torch.float32)
            return dummy, {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64)
            }

# ---------------------------------------------------------------------------
# Custom transform module that handles tensor inputs
class MyTransform(nn.Module):
    def __init__(self, min_size=320, max_size=320):
        super(MyTransform, self).__init__()
        self.min_size = min_size
        self.max_size = max_size

    def forward(self, images, targets=None):
        tensor_images = []
        for idx, img in enumerate(images):
            if not torch.is_tensor(img):
                if isinstance(img, Image.Image):
                    img = transforms.ToTensor()(img)
                else:
                    raise TypeError(f"Expected tensor or PIL Image at index {idx}, got {type(img)}")
            tensor_images.append(img)
        image_sizes = [img.shape[-2:] for img in tensor_images]
        return ImageList(torch.stack(tensor_images), image_sizes), targets

# ---------------------------------------------------------------------------
# Custom SSDLite model with modified detection head and loss
class CustomSSDLite(nn.Module):
    def __init__(self, num_classes):
        super(CustomSSDLite, self).__init__()
        base_model = ssdlite320_mobilenet_v3_large(weights='DEFAULT')
        self.backbone = base_model.backbone
        self.transform = MyTransform()
        self.num_classes = num_classes
        
        for module in base_model.backbone.features[-1].modules():
            if isinstance(module, nn.Conv2d):
                out_channels = module.out_channels
                break
        
        self.backbone_conv = nn.Conv2d(out_channels, 256, kernel_size=1)
        self.extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )
        ])
        
        self.head = SSDLiteHead(
            in_channels=256,
            num_anchors=6,
            num_classes=num_classes
        )
        
        self.anchor_generator = base_model.anchor_generator
        self.box_coder = base_model.box_coder
        
        self.score_thresh = 0.05
        self.nms_thresh = 0.5
        self.detections_per_img = 100
        
    def forward(self, images, targets=None):
        # Replace any non-tensor images with a dummy tensor or skip them.
        fixed_images = []
        for i, img in enumerate(images):
            if not isinstance(img, torch.Tensor):
                print(f"Warning: image at index {i} is not a tensor: {type(img)}; skipping this item.")
                continue
            fixed_images.append(img)
        if not fixed_images:
            raise ValueError("All images in the batch are invalid.")
        original_image_sizes = [tuple(img.shape[-2:]) for img in fixed_images]
        images, targets = self.transform(fixed_images, targets)
        
        backbone_out = self.backbone(images.tensors)
        if isinstance(backbone_out, dict):
            backbone_out = list(backbone_out.values())[0]
            
        features = OrderedDict()
        features['0'] = self.backbone_conv(backbone_out)
        
        x = backbone_out
        for i, layer in enumerate(self.extra, 1):
            x = layer(x)
            features[str(i)] = x
        
        head_outputs = self.head(features)
        anchors = self.anchor_generator(images, features)
        
        if self.training:
            return self.compute_loss(targets, head_outputs, anchors)
        else:
            return self.postprocess_detections(head_outputs, anchors, images.image_sizes, original_image_sizes)
            
    def compute_loss(self, targets, head_outputs, anchors):
        class_logits, box_regression = head_outputs
        classification_loss = F.cross_entropy(
            class_logits.view(-1, self.num_classes),
            torch.cat([t["labels"] for t in targets], dim=0)
        )
        regression_loss = F.smooth_l1_loss(
            box_regression,
            torch.cat([t["boxes"] for t in targets], dim=0),
            reduction='sum'
        ) / len(targets)
        return classification_loss + regression_loss
        
    def postprocess_detections(self, head_outputs, anchors, image_sizes, original_image_sizes):
        class_logits, box_regression = head_outputs
        pred_boxes = self.box_coder.decode(box_regression, anchors)
        pred_scores = F.softmax(class_logits, -1)
        
        pred_boxes_list = pred_boxes.split([len(a) for a in anchors])
        pred_scores_list = pred_scores.split([len(a) for a in anchors])
        
        all_boxes, all_scores, all_labels = [], [], []
        for boxes, scores, image_size in zip(pred_boxes_list, pred_scores_list, image_sizes):
            scores_per_class = scores[:, 1:]
            boxes_per_class = boxes.unsqueeze(1).expand(-1, self.num_classes - 1, -1)
            
            keep = scores_per_class > self.score_thresh
            boxes_to_keep = boxes_per_class[keep]
            scores_to_keep = scores_per_class[keep]
            labels_to_keep = torch.nonzero(keep)[:, 1] + 1
            
            keep_idx = batched_nms(
                boxes_to_keep,
                scores_to_keep,
                labels_to_keep,
                self.nms_thresh
            )
            keep_idx = keep_idx[:self.detections_per_img]
            boxes_to_keep = boxes_to_keep[keep_idx]
            scores_to_keep = scores_to_keep[keep_idx]
            labels_to_keep = labels_to_keep[keep_idx]
            
            all_boxes.append(boxes_to_keep)
            all_scores.append(scores_to_keep)
            all_labels.append(labels_to_keep)
            
        return all_boxes, all_scores, all_labels

# ---------------------------------------------------------------------------
# SSDLiteHead: Detection head for SSDLite
class SSDLiteHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.classification_head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
            )
            for _ in range(6)
        ])
        
        self.regression_head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
            )
            for _ in range(6)
        ])
        
    def forward(self, features):
        classifications, regressions = [], []
        for feature, cls_head, reg_head in zip(features.values(), 
                                               self.classification_head,
                                               self.regression_head):
            classifications.append(cls_head(feature).permute(0, 2, 3, 1).contiguous())
            regressions.append(reg_head(feature).permute(0, 2, 3, 1).contiguous())
            
        classifications = torch.cat([cls.view(cls.shape[0], -1, self.num_classes) 
                                     for cls in classifications], dim=1)
        regressions = torch.cat([reg.view(reg.shape[0], -1, 4) 
                                 for reg in regressions], dim=1)
        
        return classifications, regressions

# ---------------------------------------------------------------------------
# Custom collate function that filters out invalid items
def custom_collate_fn(batch):
    # Expect each item to be a tuple (image, target)
    valid_items = [(img, tgt) for img, tgt in batch if isinstance(img, torch.Tensor)]
    if len(valid_items) < len(batch):
        print(f"Collate: Filtered out {len(batch)-len(valid_items)} invalid items from batch.")
    images, targets = zip(*valid_items) if valid_items else ([], [])
    return list(images), list(targets)

# ---------------------------------------------------------------------------
# Training and validation functions
def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc='Training')
    
    for images, targets in progress_bar:
        for i, img in enumerate(images):
            if not torch.is_tensor(img):
                print(f"Debug: Image {i} is of type {type(img)}")
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        try:
            optimizer.zero_grad()
            loss = model(images, targets)
            if not torch.isfinite(loss):
                print('Loss is infinite or NaN. Skipping batch.')
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        except Exception as e:
            print(f"Error in training batch: {e}")
            continue
            
    return total_loss / len(progress_bar)

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(val_loader, desc='Validation')
    
    with torch.no_grad():
        for images, targets in progress_bar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            try:
                loss = model(images, targets)
                if torch.isfinite(loss):
                    total_loss += loss.item()
                progress_bar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
            except Exception as e:
                print(f"Error in validation batch: {e}")
                continue
                
    return total_loss / len(progress_bar)

# ---------------------------------------------------------------------------
# Main training loop
def main():
    TRAIN_DATASET_PATH = "/kaggle/input/rccup-voc2/ROBOCUP_OBJECTS_2024.v1-yolov3_jetson.voc/train"
    VAL_DATASET_PATH = "/kaggle/input/rccup-voc2/ROBOCUP_OBJECTS_2024.v1-yolov3_jetson.voc/val"
    NUM_EPOCHS = 50
    BATCH_SIZE = 4
    NUM_CLASSES = 23  # Your number of classes
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    train_dataset = VOCDataset(TRAIN_DATASET_PATH, transform=train_transform)
    val_dataset = VOCDataset(VAL_DATASET_PATH, transform=val_transform)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

    model = CustomSSDLite(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        try:
            train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
            val_loss = validate(model, val_loader, DEVICE)
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, 'best_model.pth')
                print(f"Saved best model with validation loss: {val_loss:.4f}")
        except Exception as e:
            print(f"Error in epoch {epoch+1}: {e}")
            continue

if __name__ == "__main__":
    main()
