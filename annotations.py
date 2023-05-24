import torch
import math

class Annotations:
    @staticmethod
    def parse_annotation(annotation_path, W, H, audio_length):
        annotations = torch.zeros((W, 1), dtype=torch.float32)

        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                time = float(parts[0])

                if time >= audio_length:
                    time = audio_length - 0.01

                index = math.floor(W * time / audio_length)
                function = parts[1]
                annotations[index] = torch.tensor([1.0]) 

        return annotations

    @staticmethod
    def reverse_annotation(annotations, audio_length):
        for probability, idx in enumerate(annotations):
            probability = float(probability)
            threshold = 0.8
            if probability > threshold:
                time = idx * audio_length / annotations.shape[0]
                print(f"Boundary: {time}")
