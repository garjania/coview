import torch
from torchvision.datasets import ImageFolder

def frame_separator(frame_path):
    return int(frame_path.split("/")[-1].split(".")[0].split("_")[-1])

class SimpleVideoDataset(ImageFolder):
    def __init__(self, root, index_slice=None, transform=None):
        # Call parent class constructor with root directory and transform
        super().__init__(
            root=root,
            transform=transform,
            target_transform=None,
        )
        if index_slice is not None:
            samples = sorted(self.samples, key= lambda x: frame_separator(x[0]))
            self.samples = samples[index_slice[0]:index_slice[1]]
        
    def __getitem__(self, idx):
        # Get the image path from parent class samples
        path = self.samples[idx][0]
        image = self.loader(path)
        frame_id = frame_separator(path)
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, frame_id
    
class SequentialVideoDataset(ImageFolder):
    def __init__(self, root, frame_len=16, index_slice=None, transform=None):
        # Call parent class constructor with root directory and transform
        super().__init__(
            root=root,
            transform=transform,
            target_transform=None,
        )
        self.frame_len = frame_len
        
        # Sort samples by frame number and optionally slice
        self.samples = sorted(self.samples, key=lambda x: frame_separator(x[0]))
        if index_slice is not None:
            self.samples = self.samples[index_slice[0]:index_slice[1]]
            
    def __getitem__(self, idx):
        # Ensure we don't go out of bounds
        if idx + self.frame_len > len(self.samples):
            raise IndexError("Sequence would extend beyond dataset bounds")
            
        frames = []
        frame_ids = []
        
        # Get the sequence of frames
        for i in range(self.frame_len):
            path = self.samples[idx + i][0]
            image = self.loader(path)
            frame_id = frame_separator(path)
            
            if self.transform is not None:
                image = self.transform(image)
                
            frames.append(image)
            frame_ids.append(frame_id)
            
        # Stack frames along a new dimension
        frames = torch.stack(frames)
        frame_ids = torch.tensor(frame_ids)
        
        return frames, frame_ids
        
    def __len__(self):
        return max(0, len(self.samples) - self.frame_len + 1)
