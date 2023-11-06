import os
import numpy as np
import librosa
from torch.utils.data import Dataset

class SpeechCommandsDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
        self.classes.sort()

        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.idx_to_class = {idx: c for c, idx in self.class_to_idx.items()} 
        
        self.class_indices = [[] for _ in range(len(self.classes))]
 
        data = []
        cur_idx = 0

        for data_class in self.classes:
            folder_path = os.path.join(folder, data_class)
            target = self.class_to_idx[data_class]

            for file_name in os.listdir(folder_path):
                path = os.path.join(folder_path, file_name)
                data.append((path, target))

                self.class_indices[target].append(cur_idx)
                cur_idx += 1

        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index]
        data = {'path': path, 'target': target}

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_classes_number(self):
        return len(self.classes)

    def get_class_from_idx(self, idx):
        if idx in self.idx_to_class.keys():
            return self.idx_to_class[idx]
        return 'unknown'
    
    def get_idx_from_class(self, c):
        if c in self.class_to_idx.keys():
            return self.class_to_idx[c]
        return -1

    def get_class_indices(self):
        return self.class_indices

    def make_weights_for_balanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

        classes_number = len(self.classes)
        classes_size = np.zeros(classes_number)

        for i in range(classes_number):
            classes_size[i] = len(self.class_indices[i])

        total_size = float(sum(classes_size))
        weight_per_class = total_size / classes_size

        weight = np.zeros(len(self))
        for idx, item in enumerate(self.data):
            weight[idx] = weight_per_class[item[1]]
        return weight

class BackgroundNoiseDataset(Dataset):
    """Dataset for silence / background noise."""

    def __init__(self, folder, transform=None, sample_rate=16000, sample_length=1):
        audio_files = [d for d in os.listdir(folder) if os.path.isfile(os.path.join(folder, d)) and d.endswith('.wav')]
        samples = []
        for f in audio_files:
            path = os.path.join(folder, f)
            s, sr = librosa.load(path, sr=sample_rate)
            samples.append(s)

        samples = np.hstack(samples)
        c = int(sample_rate * sample_length)
        r = len(samples) // c
        self.samples = samples[:r*c].reshape(-1, c)
        self.sample_rate = sample_rate
        self.transform = transform
        self.path = folder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = {'samples': self.samples[index], 'sample_rate': self.sample_rate, 'target': 1, 'path': self.path}

        if self.transform is not None:
            data = self.transform(data)

        return data
