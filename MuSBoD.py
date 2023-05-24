import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import wave
from PIL import Image
from annotations import Annotations
from torchvision.transforms import ToTensor
import torch.nn.functional as F

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
        if len(x.shape) == 5:
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
        else:
            probability_vector = []
            for window in x:
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
