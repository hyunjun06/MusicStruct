import csv
import os
import requests

def download_file(url, filename):
    if os.path.exists(filename):
        print(f"Skipping download: {filename} already exists")
        return

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {filename}: {str(e)}")

csv_file_path = "salami_dataset.csv"
output_folder = "mp3"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the first row (column headers)

    for row in csv_reader:
        song_id = row[0]  # Assuming the first column has index 0 (0-based index)
        url = row[4]  # Assuming the 5th column has index 4 (0-based index)
        filename = f"{output_folder}/{song_id}.mp3"

        download_file(url, filename)
