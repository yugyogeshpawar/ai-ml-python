# transfer_learning_example.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import os
import time

def main():
    """
    Demonstrates transfer learning by fine-tuning a pre-trained ResNet-18
    model on a small dataset (hymenoptera_data: ants vs. bees).
    """
    # --- 1. Setup: Device and Data ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Download and prepare the dataset
    # A small dataset with two classes: ants and bees
    # This is a classic transfer learning dataset.
    # Note: This will download a ~45MB file.
    print("Downloading Hymenoptera dataset...")
    # This is a bit of a hack to download and unzip the data if it doesn't exist
    if not os.path.exists('hymenoptera_data'):
        import requests, zipfile, io
        r = requests.get('https://download.pytorch.org/tutorial/hymenoptera_data.zip')
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
    print("Dataset downloaded and extracted.")

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=2)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")


    # --- 2. Load Pre-trained Model and Modify Final Layer ---
    # We will use the "feature extraction" method.
    
    # Load a pre-trained ResNet-18 model
    model = models.resnet18(pretrained=True)

    # Freeze all the parameters in the network
    for param in model.parameters():
        param.requires_grad = False

    # Get the number of input features of the final layer
    num_ftrs = model.fc.in_features

    # Replace the final fully connected layer with a new one.
    # The new layer has `num_ftrs` as input and `num_classes` as output.
    # Its parameters will have `requires_grad=True` by default.
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Move the model to the target device
    model = model.to(device)

    print("\n--- Model Architecture ---")
    print("Only the final layer's parameters will be trained.")


    # --- 3. Define Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss()

    # We only want to optimize the parameters of the final layer, which we replaced.
    # We can filter for parameters that have requires_grad = True.
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)


    # --- 4. Training and Validation Loop ---
    print("\n--- Starting Fine-Tuning ---")
    num_epochs = 5 # Set to a small number for a quick demo
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history only if in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print("--- Fine-Tuning Finished ---")


if __name__ == '__main__':
    main()
