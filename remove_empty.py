import os

dataset_directory = "dataset"

def remove_empty_folders(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):  # Check if the directory is empty
                print(f"Removing empty folder: {dir_path}")
                os.rmdir(dir_path)

remove_empty_folders(dataset_directory)

