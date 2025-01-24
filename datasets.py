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
    