import argparse
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from PIL import Image
from annotations import Annotations
from MuSBoD import MuSBoDModel
import os
import wave

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Define the MuSBoDModel class and its architecture

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the MuSBoD model on a single file")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("audio_path", type=str, help="Path to the audio file")
    args = parser.parse_args()

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = MuSBoDModel()
    model.load_state_dict(torch.load("MuSBoD.pth"))
    model.to(device)
    model.eval()

    # Process input image and obtain the predicted structural boundaries
    def process_input_image(image_path):
        image = Image.open(image_path).convert("RGB")

        image_tensor = ToTensor()(image)
        window_size = 8
        padded_image = F.pad(image_tensor, (window_size - 1, 0))
        windows = padded_image.unfold(2, window_size, 1)
        windows = windows.permute(2, 0, 3, 1)

        tensor_image = windows.to(device)
        # print(tensor_image.shape)

        # Forward pass through the model
        with torch.no_grad():
            output = model(tensor_image)

        return output

    with wave.open(args.audio_path, 'rb') as audio_file:
        num_frames = audio_file.getnframes()
        frame_rate = audio_file.getframerate()

        duration = num_frames / frame_rate

    # Load and process the input files
    image_predictions = process_input_image(args.image_path)
    print(image_predictions)
    Annotations.reverse_annotation(image_predictions, duration)


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
