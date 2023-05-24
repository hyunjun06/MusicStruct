import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

mp3_folder = "wav"
annotations_folder = "annotations"
output_folder = "dataset"
image_file_extension = "png"
batch_size = 10  # Number of files to process in each batch

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get the list of files in the mp3 folder
file_list = os.listdir(mp3_folder)

# Calculate the total number of files
total_files = len(file_list)

# Iterate through each file
for file_idx, filename in enumerate(file_list):
    try:
        song_id = os.path.splitext(filename)[0]  # Extract the song ID from the file name
        mp3_file_path = os.path.join(mp3_folder, filename)
        annotation_file_path = os.path.join(annotations_folder, song_id, "parsed", "textfile1_functions.txt")
        output_song_folder = os.path.join(output_folder, song_id)

        # Skip generating spectrogram if it already exists
        image_file_path = os.path.join(output_song_folder, f"{song_id}.{image_file_extension}")
        if os.path.exists(image_file_path):
            print(f"Skipping {song_id} - Spectrogram image already exists.")
            continue

        # Create the output folder for the song if it doesn't exist
        if not os.path.exists(output_song_folder):
            os.makedirs(output_song_folder)

        # Copy the annotation text file to the output folder
        shutil.copy(annotation_file_path, output_song_folder)

        # Load the audio file
        audio, sr = librosa.load(mp3_file_path)

        # Generate the spectrogram
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)

        # Plot and save the spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram, sr=sr, x_axis=None, y_axis=None)
        plt.axis('off')  # Turn off the axis lines and labels
        plt.margins(0)  # Remove any margins around the spectrogram
        plt.savefig(image_file_path, bbox_inches='tight', pad_inches=0)  # Save the image without extra whitespace
        plt.close()

        # Calculate and display progress percentage
        progress = (file_idx + 1) / total_files * 100
        print(f"Processed {file_idx + 1}/{total_files} files ({progress:.2f}% complete)")

        # Clean up variables to release memory
        del audio, spectrogram

    # Handle exceptions and display error messages if any
    except FileNotFoundError:
        print(f"Error: Could not find MP3 file or annotation text file for {song_id}")
    except Exception as e:
        print(f"Error: An unexpected error occurred for {song_id}\n{str(e)}")

