#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import lr_scheduler
import cv2
import pickle

# --- Configuration ---
# Set the device to use for training (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set the path to your processed dataset
data_dir = "data/processed"
# Define the number of classes for your classification problem
num_classes = 4
# Set the number of epochs for training
num_epochs = 100 # Increased epochs for more complex augmentations
# Set the learning rate for the optimizer
learning_rate = 0.001
# Set the batch size for the data loaders
batch_size = 32

# --- Data Augmentation ---
# This is where the images are treated before being fed to the network.
# A strong augmentation pipeline is key to a robust model.
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Custom Albumentations dataset to work with torchvision's ImageFolder
class AlbumentationsDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(AlbumentationsDataset, self).__init__(root)
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        image = cv2.imread(path)
        if image is None:
            print(f"Warning: Could not read image {path}. Skipping.")
            return torch.randn(3, 224, 224), target
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, target

# --- IMAGE TREATMENT & AUGMENTATION PIPELINE ---
# This pipeline defines the random transformations applied to each training image.
# This makes the model robust to variations in angle, lighting, and position.
tf_train = A.Compose([
    A.Resize(height=224, width=224),
    
    # --- NEW: More Aggressive Geometric Augmentations ---
    # This single transform applies rotation, scaling, and shifting.
    # It is crucial for making the model robust to different viewing angles.
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=45, p=0.8),
    
    # This simulates viewing the object from different perspectives, which is
    # very important for handling changes in camera angle.
    A.Perspective(scale=(0.05, 0.1), p=0.7),
    
    # --- Existing Augmentations ---
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
    A.GaussianBlur(p=0.2),
    
    # --- Normalization ---
    # This normalizes the image pixels to a standard range, which is required
    # by pre-trained models.
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

# For the validation set, we only resize and normalize. We don't apply random
# augmentations because we want a consistent, objective measure of performance.
tf_val = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])


# --- Datasets and Dataloaders ---
train_dataset = AlbumentationsDataset(root=os.path.join(data_dir, 'train'), transform=tf_train)
val_dataset = AlbumentationsDataset(root=os.path.join(data_dir, 'val'), transform=tf_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

print(f"Found {len(train_dataset)} images in the training set.")
print(f"Found {len(val_dataset)} images in the validation set.")
print(f"Classes: {train_dataset.classes}")

# --- Model ---
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)
model = model.to(device)

# --- Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# --- CNN TRAINING AND VALIDATION LOOP ---
print("\n--- Starting Training ---")
best_acc = 0.0

os.makedirs("models", exist_ok=True)
with open('models/class_names.pkl', 'wb') as f:
    pickle.dump(train_dataset.classes, f)
print("Saved class names to models/class_names.pkl")

# We loop through the dataset for a specified number of epochs.
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 10)

    # Each epoch has two phases: training and validation.
    # Training: The model learns by seeing images and adjusting its weights.
    # Validation: We test the model's performance on a separate set of images
    #             it hasn't seen before to get an unbiased measure of its accuracy.
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode. This enables features like Dropout.
            loader = train_loader
        else:
            model.eval()   # Set model to evaluate mode. This disables features like Dropout
                           # for consistent, deterministic predictions.
            loader = val_loader

        running_loss = 0.0
        running_corrects = 0

        # Iterate over batches of data from the dataloader.
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # --- Key Step: Zero the gradients ---
            # Before each batch, we clear any gradients from the previous step.
            optimizer.zero_grad()

            # --- Key Step: Forward Pass ---
            # We run the model on the input images. `torch.set_grad_enabled` ensures
            # that we only calculate gradients during the training phase.
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1) # Get the class with the highest score
                loss = criterion(outputs, labels) # Calculate the error (loss)

                # --- Key Step: Backward Pass (Training Only) ---
                # If we are in the training phase, we perform backpropagation.
                # This calculates how much each model weight contributed to the error.
                if phase == 'train':
                    loss.backward()
                    # The optimizer then updates the weights to reduce the error.
                    optimizer.step()

            # --- Statistics ---
            # We track the loss and number of correct predictions to monitor performance.
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        if phase == 'train':
            scheduler.step() # Adjust the learning rate if needed.

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = running_corrects.double() / len(loader.dataset)

        print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # --- Key Step: Model Checkpointing ---
        # After each epoch, if the model's performance on the *validation set*
        # is the best we've seen so far, we save its weights.
        # We use validation accuracy to avoid saving a model that is overfit.
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), "models/best_resnet18.pth")
            print("✅  New best model saved!")

print("\n--- Training done ---")
print(f"Best validation accuracy: {best_acc:.4f}")

