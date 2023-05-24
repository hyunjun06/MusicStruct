import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from MuSBoD import MuSBoDModel, MuSBoDDataset
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuSBoD Training Script. Requires output file path.")
    parser.add_argument("output_file", type=str, help="Path to save the trained model")

    args = parser.parse_args()
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the dataset and dataloader
    dataset = MuSBoDDataset("dataset")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the model
    model = MuSBoDModel().to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
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
    torch.save(model.state_dict(), args.output_file)
