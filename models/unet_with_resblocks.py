import torch
import torch.nn as nn
#import math
import torch.nn.functional as F

'''class ECA(nn.Module):
    def __init__(self,channels,gamma=2, b=1):
        super().__init__()
        k = int(abs(math.log2(channels)*gamma + b))
        k = max(k,3)
        if k % 2 == 0: k +=1
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias = False)
        self.sigmoid= nn.Sigmoid()

    def forward(self,x):
            b,c,_,_= x.size()
            y = self.pool(x).view(b, 1, c)
            y = self.conv1d(y).view(b, c, 1, 1)
            y = self.sigmoid(y)
            return x * y.expand_as(x)
'''                
        
'''class RMSNorm (nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.norm_factor= channels ** 0.5
        self.channel_weight= nn.Parameter(torch.ones(1,channels,1,1))

    def forward(self,x):
        x= F.normalize(x,dim=1)
        return x*self.channel_weight*self.norm_factor'''
class RMSNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm=nn.RMSNorm(channels)
    def forward(self,x):
        x=x.permute(0,2,3,1)
        x=self.norm(x)
        x=x.permute(0,3,1,2)
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1),
            RMSNorm(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

    def forward (self,x):
        return self.layers(x)   


class ResBlock(nn.Module):
    def __init__(self,in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.block1=Block(in_channels, out_channels, dropout=dropout)
        self.block2=Block(out_channels, out_channels, dropout=0.0)
        #self.eca= ECA(out_channels)

        #projection case
        if in_channels != out_channels:
            self.shortcut=nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else: 
            self.shortcut=nn.Identity()  


    def forward(self,x):
        h=self.block1(x)
        h=self.block2(h)
        #h=self.eca(h)
        return self.shortcut(x) + h 

class Downsample(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.conv=nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1 )

    def forward(self,x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self,in_channels):
        super().__init__() 
        self.upsample= nn.Upsample(scale_factor=2, mode="nearest")
        self.conv= nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self,x):
        x=self.upsample(x)
        x=self.conv(x)
        return x   

class DownLevel(nn.Module):
    def __init__(self, in_channels, out_channels, nbr_rb, dropout):
        super().__init__()
        self.blocks=nn.ModuleList()
        current_channels= in_channels
        for i in range(nbr_rb):
            self.blocks.append(ResBlock(current_channels, out_channels, dropout=dropout))
            current_channels= out_channels
        self.downsample= Downsample(out_channels)
    def forward(self,x):
        for block in self.blocks:
            x= block(x) 
        encoder_features = x
        x=self.downsample(x)
        return x, encoder_features

class Uplevel(nn.Module):
    def __init__(self, in_channels, encoder_channels, out_channels, nbr_rb, dropout):
        super().__init__()
        self.upsample=Upsample(in_channels)
        self.blocks= nn.ModuleList()
        #nbr of channels after concat
        current_channels= in_channels + encoder_channels
        for i in range (nbr_rb):
            self.blocks.append(ResBlock(current_channels,out_channels, dropout=dropout))
            current_channels= out_channels
    def forward(self, x, encoder_features):
        x=self.upsample(x)
        x=torch.cat([x,encoder_features], dim=1)
        for block in self.blocks:
            x=block(x)
        return x    

class UNetRes(nn.Module):
    def __init__(self,in_channels =2, out_channels=1, base_channels=128, channels_multiplier=(1,2,4),
                 nbr_rb=2, dropout=0.1, ):
        super().__init__()
        channels=[]
        for m in channels_multiplier: channels.append(base_channels*m)
        self.in_conv= nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        current_channels= channels[0]
        encoder_channels= channels[:-1]
        bottleneck_channels=channels[-1]
        #encoder
        self.down_levels= nn.ModuleList()
        for level_out_ch in encoder_channels:
            down_level= DownLevel(current_channels, level_out_ch, nbr_rb, dropout)
            self.down_levels.append(down_level)
            current_channels= level_out_ch           
        #bottleneck
        self.bottleneck_blocks= nn.ModuleList()
        for i in range (nbr_rb):
            self.bottleneck_blocks.append(ResBlock(current_channels,bottleneck_channels, dropout=dropout))
            current_channels= bottleneck_channels
        #decoder
        self.up_levels=nn.ModuleList()
        for encoder_channels in list(reversed(encoder_channels)):
            up_level= Uplevel(current_channels, encoder_channels, encoder_channels, nbr_rb,dropout)
            self.up_levels.append(up_level)
            current_channels=encoder_channels

        self.out_normalization= RMSNorm(current_channels)
        self.activation=nn.SiLU()
        self.out_conv=nn.Conv2d(current_channels,out_channels,kernel_size=3, padding=1)
        #self.output_activation= nn.Sigmoid()

    def forward(self, y, mask):
        if mask.dim()==3:
            mask= mask.unsqueeze(0)
        mask_batch=mask.expand_as(y)
        model_input=torch.cat([y,mask_batch], dim=1)  

        x=self.in_conv(model_input)
        encoder_features_list=[]
        for down_level in self.down_levels:
            x,encoder_features=down_level(x)
            encoder_features_list.append(encoder_features)
        for block in self.bottleneck_blocks:
            x=block(x)
        for up_level in self.up_levels:
            encoder_features=encoder_features_list.pop()
            x=up_level(x,encoder_features)   
        x=self.out_normalization(x)
        x=self.activation(x)
        predicted = self.out_conv(x)
        #predicted= self.output_activation(predicted)

        reconstruction=y*mask_batch + predicted*(1-mask_batch)
        return reconstruction              