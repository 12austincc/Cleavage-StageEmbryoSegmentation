import torch.nn as nn
import torch

class UpBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
    
    def forward(self,x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class SemanticDecoder(nn.Module):
    def __init__(self,num_classes=2):
        super().__init__()
        self.up1 = UpBlock(256,64)
        self.up2 = UpBlock(64,16)
        self.up3 = UpBlock(16,4)
        self.up4 = UpBlock(4,num_classes)
        

    def forward(self,x):
        '''
        (1,256,64,64)->
        (1,64,128,128)->(1,16,256,256)
        ->(1,4,512,512)->(1,1,1024,1024)
        '''
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return x
    

if __name__ == '__main__':
    x =torch.rand(1,256,64,64)
    d = SemanticDecoder(num_classes=3)
    y =d(x)
    prediction = torch.argmax(y,dim=1)
    print(y.shape)
    print(prediction.shape)