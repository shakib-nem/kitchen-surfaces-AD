import torch
from torchvision import models
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor

class _CutPasteNetBase(nn.Module):
    # forward outputs: logits
    def __init__(self, encoder = 'resnet50', pretrained = True, dims = [2048,512,512,512,512,512,512,512,128], num_class = 2):
        super().__init__()
        self.encoder = getattr(models, encoder)(pretrained = pretrained)
        last_layer= list(self.encoder.named_modules())[-1][0].split('.')[0]
        setattr(self.encoder, last_layer, nn.Identity())
        
        #when you will use resnet50, uncomment this part and comment the below part
        #resnet50
        proj_layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            proj_layers.append(nn.Linear(d_in, d_out, bias=False))
            proj_layers.append(nn.BatchNorm1d(d_out))
            proj_layers.append(nn.ReLU(inplace=True))
        self.head = nn.Sequential(*proj_layers)
        self.out = nn.Linear(dims[-1], num_class)
        
        
        '''
        ##when you will use resnet18, uncomment this part and comment the above part
        #resnet18
        proj_layers = []
        for d in dims[:-1]:
            proj_layers.append(nn.Linear(d,d, bias=False))
            proj_layers.append((nn.BatchNorm1d(d)))
            proj_layers.append(nn.ReLU(inplace=True))
        embeds = nn.Linear(dims[-2], dims[-1], bias=num_class > 0)
        proj_layers.append(embeds)
        self.head = nn.Sequential(
            *proj_layers
        )
        self.out = nn.Linear(dims[-1], num_class)
        '''
        
    def forward(self, x):
        features = self.encoder(x)
        embeds = self.head(features)
        logits = self.out(embeds)
        return logits

    def freeze(self, layer_name):
        #freeze encoder until layer_name
        check = False
        for name, param in self.encoder.named_parameters():
            if name == layer_name:
                check = True 
            if not check and param.requires_grad != False:
                param.requires_grad = False
            else:
                param.requires_grad = True
                
    def create_graph_model(self,):
        return create_feature_extractor(model=self, return_nodes=["head", "out"])
    
    
class CutPasteNet(_CutPasteNetBase):
    # forward outputs:  (logits, embeds)
    #when you will use resnet18, change the first value of dims to 512 and encoder to "resnet18"
    def __init__(self, encoder='resnet50', pretrained=True, dims=[2048, 512, 512, 512, 512, 512, 512, 512, 128], num_class=2): #first value was 512, changed it to make it work with resnet50
        super().__init__(encoder, pretrained, dims, num_class)
        return
    
    def forward(self, x):
        features = self.encoder(x)
        embeds = self.head(features)
        logits = self.out(embeds)
        return (logits, embeds)
    
if __name__ == '__main__':
    model = CutPasteNet()
    print(model)

