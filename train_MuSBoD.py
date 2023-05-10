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
from annotations import Annotations

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Define the MuSBoDModel
class MuSBoDModel(nn.Module):
    def __init__(self):
        super(MuSBoDModel, self).__init__()
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

# Define the MuSBoDDataset
class MuSBoDDataset(Dataset):
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
        annotation = Annotations.parse_annotation(annotation_path, image.width, image.height, duration)

        image_tensor = ToTensor()(image)
        window_size = 8
        padded_image = F.pad(image_tensor, (window_size - 1, 0))
        windows = padded_image.unfold(2, window_size, 1)
        windows = windows.permute(2, 0, 3, 1)

        return windows, annotation

if __name__ == "__main__":
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the dataset and dataloader
    dataset = MuSBoDDataset("dataset")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the model
    model = MuSBoDModel().to(device)

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
    torch.save(model.state_dict(), "MuSBoD.pth")

# # Load the trained model
# model = MuSBoDModel()
# model.load_state_dict(torch.load("MuSBoD.pth"))
# 
# # Set the model to evaluation mode
# model.eval()
# 
# # Process input
# # Process input image and obtain the predicted structural boundaries
# def process_input_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     tensor_image = ToTensor()(image).unsqueeze(0).to(device)
# 
#     # Forward pass through the model
#     with torch.no_grad():
#         output = model(tensor_image)
# 
#     # Convert output to binary predictions (0s and 1s)
#     predictions = torch.sigmoid(output)
#     binary_predictions = (predictions > 0.5).squeeze().cpu().numpy()
# 
#     return binary_predictions
# 
# # Example usage
# image_path = "path_to_input_image.png"
# predictions = process_input_image(image_path)
# 
# # Convert specific time of the song to index in the 1D tensor output
# def time_to_index(time, sr, hop_length):
#     return int(time / (hop_length / sr))
# 
# # Example usage
# time = 96.671723356  # Specific time of the song
# sr = 22050  # Sample rate
# hop_length = 512  # Hop length used in spectrogram calculation
# index = time_to_index(time, sr, hop_length)
# print(f"Index corresponding to time {time}s: {index}")
