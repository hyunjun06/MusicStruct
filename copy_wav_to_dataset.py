import os
import shutil

dataset_folder = "dataset"
wav_folder = "wav"

# Iterate through all folders in the dataset directory
for folder_name in os.listdir(dataset_folder):
    folder_path = os.path.join(dataset_folder, folder_name)

    # Check if the path corresponds to a directory
    if os.path.isdir(folder_path):
        # Create the folder in the dataset directory if it doesn't exist
        output_folder_path = os.path.join(dataset_folder, folder_name)
        os.makedirs(output_folder_path, exist_ok=True)

        # Construct the source file path
        source_file_path = os.path.join(wav_folder, f"{folder_name}.wav")

        # Check if the source file exists
        if os.path.exists(source_file_path):
            # Copy the source file to the output folder
            output_file_path = os.path.join(output_folder_path, f"{folder_name}.wav")
            shutil.copy(source_file_path, output_file_path)
            print(f"Copied file from {source_file_path} to {output_file_path}")
        else:
            print(f"Source file not found: {source_file_path}")
    else:
        print(f"Not a directory: {folder_path}")

