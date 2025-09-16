from torch.utils.data import Dataset
from torchvision import transforms, utils, datasets


class SubsetImageFolder(Dataset):
    def __init__(self, root, classes_file, transform=None):
        # Read the class names from the file
        with open(classes_file, 'r') as f:
            selected_classes = [line.strip() for line in f.readlines()]

        # Create a set for O(1) lookup
        self.selected_classes = set(selected_classes)

        # Initialize the ImageFolder dataset
        self.dataset = datasets.ImageFolder(root=root, transform=transform)

        # Create a mapping from original indices to new indices
        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(selected_classes)
        }

        # Filter indices based on selected classes
        self.indices = []
        for idx, (_, label) in enumerate(self.dataset.samples):
            original_class = self.dataset.classes[label]
            if original_class in self.selected_classes:
                self.indices.append(idx)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the original sample
        image, label = self.dataset[self.indices[idx]]

        # Map the original label to new label index
        original_class = self.dataset.classes[label]
        new_label = self.class_to_idx[original_class]

        return image, new_label
