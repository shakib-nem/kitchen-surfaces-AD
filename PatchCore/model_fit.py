
from torch.utils.data import DataLoader
from BurgerData import BurgerData
from ind_knn_ad.indad.models import PatchCore
import torchvision
import torch
import torch.nn as nn

#for white
mean = [0.5815, 0.5940, 0.5015]
std = [0.2716, 0.2812, 0.2710]

#for white with edges
#mean = [0.6384, 0.6557, 0.5500]
#std = [0.2846, 0.2897, 0.2772]


trans = torchvision.transforms.Normalize(mean=mean,std=std)

data = BurgerData(imgSize=224, stride=512, image_folder=r"/home/shn/data/white/train", transform=trans)
loader = DataLoader(data, batch_size=64, shuffle=True,num_workers=16)

#train the model
model = PatchCore( backbone_name="resnet50")
model.to("cuda:0")

# feed healthy dataset
model.fit(loader)

#save the model
torch.save(model, '50white_blur')
