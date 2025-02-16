import os
import torch
import torchvision
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
import cv2
import torch.optim as optim
import torch.utils.data as data

# Define image preprocessing transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((320, 320)),  # Resize to model's expected input size
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
        self.class_dict = {name: i+1 for i, name in enumerate(self.class_names)}

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
DATASET_PATH = '/kaggle/input/rccup-voc2/ROBOCUP_OBJECTS_2024.v1-yolov3_jetson.voc/train'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset and Loader
train_dataset = VOCDataset(DATASET_PATH)
train_loader = data.DataLoader(
    train_dataset, 
    batch_size=4, 
    shuffle=True, 
    drop_last=True,  # Ensure consistent batch sizes
    collate_fn=lambda x: tuple(zip(*x))
)

# Model Setup
num_classes = 23
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(num_classes=num_classes)
model = model.to(device)

# Training Loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10

for epoch in range(num_epochs):
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

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# Save Model
torch.save(model.state_dict(), "ssdlite_mobilenet_v3_large_voc.pth")
