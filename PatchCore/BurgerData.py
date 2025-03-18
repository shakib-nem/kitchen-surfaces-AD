import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy
import time
import cv2
class BurgerData(Dataset):
    """
        Initializes the BurgerData dataset.

        Parameters:
        - imgSize: The size (width and height) of the square patch to be cropped from the images.
        - stride: The step size used when sliding over the image to generate patches.
        - image_folder: Path to the folder containing the images.
        - transform: Optional torchvision transformation to apply to each patch.
    """
    def __init__(self, imgSize: int, stride: int, image_folder: str, transform= None) -> None:
        super().__init__()

        self.imgSize = imgSize
        self.stride = stride
        self.image_folder = image_folder
        self.index2position = self._generate_index2position()
        self.transform = transform
    
    def _generate_index2position(self): #this function generates the location of the patches in the image
        """
        Generates a list of valid patch positions from images in the folder.
        Each entry in the list is of the form [filename, (x, y)], where (x, y)
        represents the top-left corner of the patch within the image.

        Only patches that are not too dark (mean pixel value > 31) are included.
        """
        
        index2position = [] 
        supported_extensions = ('_1.jpg')

        for filename in os.listdir(self.image_folder):
            if filename.endswith(supported_extensions):
                img = Image.open(f"{self.image_folder}/{filename}")
                
                width, height = img.size 

                #to throw away smaller patches
                patches_per_row=((width - self.imgSize) // self.stride + 1)*self.stride #get the number of patches in a row
                patches_per_column=((height - self.imgSize) // self.stride + 1)*self.stride #get the number of patches in a column
                
                for y in range(0, patches_per_column, self.stride):
                    for x in range(0, patches_per_row, self.stride):
                        subimage = img.crop((x, y, x + self.imgSize, y + self.imgSize))  
                        if numpy.mean(numpy.array(subimage))>31:
                            index2position.append([filename, (x, y)]) # only append if not black  
                           
        return index2position

    def __len__(self) -> int:
        return len(self.index2position)
    
    def _reduce_noise(self, image):
        """
        Apply noise reduction using OpenCV's fastNlMeansDenoisingColored.
        :image: PIL.Image - Input image.
        :return: PIL.Image - Noise-reduced image.
        """
        # Convert PIL image to OpenCV format (numpy array)
        cv_img = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)

        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, 3, 3, 7, 21)

        # Convert back to PIL image
        return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
    
    def __getitem__(self, index) -> torch.Tensor:
        """
        Retrieves a patch from the dataset based on the provided index.
        Applies any necessary transformations and returns a tuple containing
        the image tensor and the label

        Parameters:
        - index: Index of the patch to retrieve.

        Returns:
        - A tuple (patch_tensor, 0) where patch_tensor is the transformed image patch.
        """
        file, (x, y) = self.index2position[index]  
        img = Image.open(f"{self.image_folder}/{file}")
        patch = img.crop((x, y, x + self.imgSize, y + self.imgSize))
        #patch = self._reduce_noise(patch) #uncomment to apply for meodels needing noise reduction (blur) 

        if self.transform:
            transform_compose = transforms.Compose([transforms.ToTensor(), self.transform])
            patch_tensor = transform_compose(patch)
            
        else:
            #Only apply ToTensor if no other transform is applied
            transformit=transforms.ToTensor()
            patch_tensor = transformit(patch)
        
        return patch_tensor, 0 
            
    
def calculate_mean_std(data):
    """
    Calculates the mean and standard deviation of all image tensors in the dataset
    for normalizing the data.

    Parameters:
    - data: An iterable (or dataset) where each item is a tuple (image_tensor, label).

    Returns:
    - A tuple (mean, std) representing the mean and standard deviation computed
      over all channels and spatial dimensions.
    """      
    image_tensor = torch.stack([item[0] for item in data])
    mean = torch.mean(image_tensor, dim=(0, 2, 3))
    std = torch.std(image_tensor, dim=(0, 2, 3)) 
    return mean, std
    


if __name__ == "__main__":
    
    #white
    mean=[0.5815, 0.5940, 0.5015]
    std=[0.2716, 0.2812, 0.2710]
        
    #white_with_edges
    #mean=[0.6384, 0.6557, 0.5500]
    #std=[0.2846, 0.2897, 0.2772]
    
    #transform =  transforms.Normalize(mean=mean, std=std)
    #data = BurgerData(imgSize=224, stride=448, image_folder=r"/home/shn/data/white_with_edges/train", transform=transform)  
   
    data = BurgerData(imgSize=224, stride=448, image_folder =r"/home/shn/data/white_with_edges/train") #for white with edges
    print(len(data))
    