import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
import os
import wave
import math
import numpy as np
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3, 4"

# Define the MusicStructModel
class MusicStructModel(nn.Module):
    def __init__(self):
        super(MusicStructModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 9))
        self.pool = nn.MaxPool2d(kernel_size=(1, 3))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(6, 10))
        self.fc1 = nn.Linear(32 * 30, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        output = []
        for data in x:
            probability_vector = []
            for window in data:
                window = self.pool(torch.relu(self.conv1(window)))
                window = self.pool(torch.relu(self.conv2(window)))
                window = window.view(-1)  # Flatten the tensor
                window = torch.relu(self.fc1(window))
                window = self.fc2(window)
                probability_vector.append(window)
            probability_vector = torch.stack(probability_vector, dim=0)
            output.append(probability_vector)
        output = torch.stack(output, dim=0)
        return output

# Define the MusicStructDataset
class MusicStructDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_files[idx], self.image_files[idx] + ".png")
        annotation_path = os.path.join(self.root_dir, self.image_files[idx],  "textfile1_functions.txt")
        audio_path = os.path.join(self.root_dir, self.image_files[idx], self.image_files[idx] + ".wav")

        with wave.open(audio_path, 'rb') as audio_file:
            num_frames = audio_file.getnframes()
            frame_rate = audio_file.getframerate()

            duration = num_frames / frame_rate

        image = Image.open(image_path).convert("RGB")
        annotation = self.parse_annotation(annotation_path, image.width, image.height, duration)

        image_tensor = ToTensor()(image)
        window_size = 8
        padded_image = F.pad(image_tensor, (window_size - 1, 0))
        windows = padded_image.unfold(2, window_size, 1)
        windows = windows.permute(2, 0, 3, 1)

        return windows, annotation

    def parse_annotation(self, annotation_path, W, H, audio_length):
        annotations = torch.zeros((W, 1), dtype=torch.float32)

        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                time = float(parts[0])

                if time >= audio_length:
                    time = audio_length - 0.01

                index = math.floor(W * time / audio_length)
                function = parts[1]
                annotations[index] = torch.tensor([1.0]) 

        return annotations

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the dataset and dataloader
dataset = MusicStructDataset("dataset")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model and wrap it with DataParallel
model = MusicStructModel().to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
    for images, annotations in progress_bar:
        images = images.to(device)
        annotations = annotations.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, annotations)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix(loss=loss.item())
        progress_bar.update()

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    # Close the progress bar
    progress_bar.close()

print("Training finished.")

# Save the trained model
torch.save(model.state_dict(), "music_struct_model.pth")

