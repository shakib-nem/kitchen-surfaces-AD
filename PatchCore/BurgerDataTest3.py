import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import json  # Import json to load the JSON file
import cv2

class BurgerData(Dataset):
    def __init__(self, imgSize: int, stride: int, image_folder: str, json_file: str, transform=None) -> None:
        super().__init__()

        """
        Initializes the BurgerDataTest dataset.

        Parameters:
        - imgSize: The size (width and height) of the square patch to be cropped from the images.
        - stride: The step size used when sliding over the image to generate patches.
        - image_folder: Path to the folder containing the images.
        - json_file: JSON file with metadata (labels and borders) for each image.
        - transform: Optional torchvision transformation to apply to each patch.
        """
        
        self.imgSize = imgSize
        self.stride = stride
        self.image_folder = image_folder
        self.transform = transform

        # Load the JSON data
        with open(json_file, 'r') as f:
            self.json_data = json.load(f)

        self.index2position = self._generate_index2position()

    def _generate_index2position(self):
        """
        For each image:
          - Check that the image exists in the JSON metadata.
          - Retrieve the border (region of interest) and label bounding boxes.
          - Generate a grid of potential patch positions within the border.
          - Adjust the grid to include edge cases.
          - Filter out patches that overlap with any label.
        Returns a list mapping dataset indices to (filename, (x, y)) patch positions.
        """
        
        index2position = []
        supported_extensions = ('_1.jpg')

        for filename in os.listdir(self.image_folder):
            if filename.endswith(supported_extensions):
                img = Image.open(f"{self.image_folder}/{filename}")

                if filename in self.json_data:
                    labels = self.json_data[filename]['labels']
                    borders = self.json_data[filename]['borders']
                else:
                    print(f"{filename} not found in JSON data. Available keys: {list(self.json_data.keys())}...")
                    continue

                # Assuming only one border rectangle per image
                border_rect = borders[0]
                bx, by, bw, bh = border_rect

                # Prepare labels rectangles
                labels_rects = np.array(labels)  # Convert to NumPy array for vectorized operations
                labels_x1 = labels_rects[:, 0]
                labels_y1 = labels_rects[:, 1]
                labels_x2 = labels_x1 + labels_rects[:, 2]
                labels_y2 = labels_y1 + labels_rects[:, 3]

                # Generate grid of patch positions within the borders
                x_coords = np.arange(bx, bx + bw - self.imgSize + 1, self.stride)
                y_coords = np.arange(by, by + bh - self.imgSize + 1, self.stride)
                x_grid, y_grid = np.meshgrid(x_coords, y_coords)
                patch_positions = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)

                # Add edge cases (right and bottom edges)
                if bw % self.stride != 0:
                    right_edge_x = np.full((y_coords.shape[0],), bx + bw - self.imgSize)
                    patch_positions = np.vstack([patch_positions, np.stack([right_edge_x, y_coords], axis=1)])
                if bh % self.stride != 0:
                    bottom_edge_y = np.full((x_coords.shape[0],), by + bh - self.imgSize)
                    patch_positions = np.vstack([patch_positions, np.stack([x_coords, bottom_edge_y], axis=1)])

                # Add bottom-right corner
                if bw % self.stride != 0 and bh % self.stride != 0:
                    patch_positions = np.vstack([patch_positions, [bx + bw - self.imgSize, by + bh - self.imgSize]])

                # Filter patches
                for px, py in patch_positions:
                    patch_x1 = px
                    patch_y1 = py
                    patch_x2 = px + self.imgSize
                    patch_y2 = py + self.imgSize

                    # Check for overlaps with any labels
                    overlaps = (
                        (patch_x1 < labels_x2) &
                        (patch_x2 > labels_x1) &
                        (patch_y1 < labels_y2) &
                        (patch_y2 > labels_y1)
                    )

                    if not overlaps.any():
                        index2position.append([filename, (int(patch_x1), int(patch_y1))])

        return index2position



    def __len__(self) -> int:
        return len(self.index2position)

    def _reduce_noise(self, image):
        """
        Apply noise reduction using OpenCV's fastNlMeansDenoisingColored.
        :image: PIL.Image - Input image.
        :return: PIL.Image - Noise-reduced image.
        """
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, 3, 3, 7, 21)
        return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))

    def __getitem__(self, index) -> torch.Tensor:
        """
        Retrieve a patch:
          - Use the precomputed position to crop the patch from the image.
          - Optionally apply noise reduction.
          - Apply the provided transform to convert the patch to a tensor.
          - Load a corresponding groundtruth image, crop it, and determine a binary label (anomaly = 1, normal = 0).
        Returns a tuple (patch_tensor, label).
        """
        file, (x, y) = self.index2position[index]
        img_path = os.path.join(self.image_folder, file)
        img = Image.open(img_path)
        patch = img.crop((x, y, x + self.imgSize, y + self.imgSize))
        #patch = self._reduce_noise(patch) #uncomment to apply for meodels needing noise reduction (blur)

        if self.transform:
            transform_compose = transforms.Compose([transforms.ToTensor(), self.transform])
            patch_tensor = transform_compose(patch)
            
            # Load the corresponding groundtruth image and crop the same patch area.
            groundtruth_img_path = os.path.join(self.image_folder, file[:-4] + '_groundtruth.png')
            groundtruth_img = Image.open(groundtruth_img_path)
            groundtruth_img = groundtruth_img.convert('L')
            groundtruth_patch = groundtruth_img.crop((x, y, x + self.imgSize, y + self.imgSize))
            groundtruth_patch_np = np.array(groundtruth_patch)
            
            # Create a binary label based on the average intensity of the groundtruth patch.
            label = 1 if np.mean(groundtruth_patch_np) > 0 else 0
        else:
            raise ValueError("Transform is not provided. Please provide a valid transform.")

        return patch_tensor, label


def calculate_mean_std(data):
    image_tensor = torch.stack([item[0] for item in data])
    mean = torch.mean(image_tensor, dim=(0, 2, 3))
    std = torch.std(image_tensor, dim=(0, 2, 3))
    return mean, std


if __name__ == "__main__":
    mean = [0.6047, 0.6183, 0.5254]
    std = [0.2732, 0.2820, 0.2739]
    transform = transforms.Normalize(mean=mean, std=std)
    data = BurgerData(
        imgSize=244,
        stride=112,
        image_folder="/home/shn/data/test/white",
        json_file='/home/shn/PatchCore/white_coords.json',
        transform=transform
    )

    print("Total number of patches:", len(data))
