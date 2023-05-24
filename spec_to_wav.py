import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io import wavfile


def generate_audio_from_spectrogram(png_file, output_file):
    # Load the PNG spectrogram image
    spectrogram = plt.imread(png_file)

    # Convert the PNG image to magnitude spectrogram
    magnitude = librosa.db_to_amplitude(spectrogram)

    # Reconstruct the complex spectrogram
    complex_spectrogram = magnitude * np.exp(1j * np.angle(spectrogram))

    # Invert the complex spectrogram to a time-domain signal using the Griffin-Lim algorithm
    signal_data = librosa.griffinlim(complex_spectrogram)

    # Normalize the signal to the range [-1, 1]
    signal_data /= np.max(np.abs(signal_data))

    # Rescale the signal to 16-bit integers
    signal_data = (signal_data * 32767).astype(np.int16)

    # Save the audio signal as a WAV file
    wavfile.write(output_file, 22050, signal_data)

# Check if the command-line arguments are provided
if len(sys.argv) < 3:
    print("Usage: python generate_audio.py input.png output.wav")
    sys.exit(1)

# Get the input PNG file path and output WAV file path from the command-line arguments
input_png = sys.argv[1]
output_wav = sys.argv[2]

# Generate audio from the spectrogram and save it as a WAV file
generate_audio_from_spectrogram(input_png, output_wav)

print(f"Audio generated from {input_png} and saved as {output_wav}")

