import argparse
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

mp3_folder = "mp3"
annotations_folder = "annotations"
output_folder = "dataset"
image_file_extension = "png"
batch_size = 10  # Number of files to process in each batch

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get the list of files in the mp3 folder
file_list = os.listdir(mp3_folder)

# Calculate the total number of batches
num_batches = len(file_list) // batch_size
if len(file_list) % batch_size != 0:
    num_batches += 1

# ANSI escape sequence for green color
green_color_code = "\033[92m"

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate spectrograms from MP3 files.")
parser.add_argument("--max-files", type=int, help="Maximum number of files to process")
args = parser.parse_args()

max_files = args.max_files

# Limit the file list if max_files is specified
if max_files is not None and max_files < len(file_list):
    file_list = file_list[:max_files]

# Iterate through each batch of files
for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = (batch_idx + 1) * batch_size
    batch_files = file_list[start_idx:end_idx]

    # Process each file in the batch
    for filename in batch_files:
        try:
            song_id = os.path.splitext(filename)[0]  # Extract the song ID from the file name
            mp3_file_path = os.path.join(mp3_folder, filename)
            annotation_file_path = os.path.join(annotations_folder, song_id, "parsed", "textfile1_functions.txt")
            output_song_folder = os.path.join(output_folder, song_id)

            # Create the output folder for the song if it doesn't exist
            if not os.path.exists(output_song_folder):
                os.makedirs(output_song_folder)

            # Copy the annotation text file to the output folder
            shutil.copy(annotation_file_path, output_song_folder)

            # Convert MP3 to WAV using an external tool or library
            wav_file_path = os.path.join(output_song_folder, f"{song_id}.wav")
            # Replace the command below with the appropriate command for MP3 to WAV conversion
            os.system(f"ffmpeg -i {mp3_file_path} {wav_file_path}")

            # Load the WAV file
            sample_rate, audio = wavfile.read(wav_file_path)

            # Generate the spectrogram
            _, _, spectrogram = spectrogram(audio, fs=sample_rate)

            # Define the image file path
            image_file_path = os.path.join(output_song_folder, f"{song_id}.{image_file_extension}")

            # Plot and save the spectrogram
            plt.figure(figsize=(10, 4))
            plt.imshow(spectrogram, aspect='auto', cmap='jet', origin='lower')
            plt.axis('off')  # Turn off the axis lines and labels
            plt.margins(0)  # Remove any margins around the spectrogram
            plt.savefig(image_file_path, bbox_inches='tight', pad_inches=0)  # Save the image without extra whitespace
            plt.close()

            # Print the message in green color
            print(f"{green_color_code}Generated spectrogram for {song_id} and saved to {image_file_path}\033[0m")

            # Clean up the temporary WAV file
            os.remove(wav_file_path)

        # Handle exceptions and display error messages if any
        except FileNotFoundError:
            print(f"Error: Could not find MP3 file or annotation text file for {song_id}")
        except Exception as e:
            print(f"Error: An unexpected error occurred for {song_id}\n{str(e)}")
