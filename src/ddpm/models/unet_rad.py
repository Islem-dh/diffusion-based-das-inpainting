import torch
import torch.nn as nn
from ddpm.utils.spatial_noise_embedding import SpatialNoiseEmbedding


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
    def __init__(self,in_channels, out_channels, noise_emb_dim, dropout=0.0 ):
        super().__init__()
        self.block1=Block(in_channels, out_channels, dropout=dropout)
        self.block2=Block(out_channels, out_channels, dropout=0.0)
        #self.eca= ECA(out_channels)

       
        self.noise_projection= nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(noise_emb_dim, out_channels,kernel_size=1)
        )
        

        #projection case
        if in_channels != out_channels:
            self.shortcut=nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else: 
            self.shortcut=nn.Identity()  


    def forward(self,x,noise_emb):
        h=self.block1(x)
        if noise_emb.shape[-2:] != h.shape[-2:]:
            noise_emb_resized=nn.functional.interpolate(noise_emb, size=h.shape[-2:], mode="nearest")
        else : noise_emb_resized= noise_emb

        #if noise_features.shape[-2:] != h.shape[-2:]:
           # noise_features=nn.functional.interpolate(noise_features, size=h.shape[-2:], mode="nearest")

        noise_features = self.noise_projection(noise_emb_resized)
        h=h+noise_features
        h=self.block2(h)
        
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
    def __init__(self, in_channels, out_channels, nbr_rb,noise_emb_dim, dropout):
        super().__init__()
        self.blocks=nn.ModuleList()
        current_channels= in_channels

        for i in range(nbr_rb):
            self.blocks.append(
                ResBlock(current_channels, out_channels,noise_emb_dim=noise_emb_dim, dropout=dropout)
            )

            current_channels= out_channels
        self.downsample= Downsample(out_channels)

    def forward(self,x, noise_emb):
        for block in self.blocks:
            x= block(x,noise_emb) 
        encoder_features = x
        x=self.downsample(x)
        return x, encoder_features

class Uplevel(nn.Module):
    def __init__(self, in_channels, encoder_channels, out_channels, nbr_rb, noise_emb_dim, dropout):
        super().__init__()
        self.upsample=Upsample(in_channels)
        self.blocks= nn.ModuleList()
        #nbr of channels after concat
        current_channels= in_channels + encoder_channels
        for i in range (nbr_rb):
            self.blocks.append(
                ResBlock(current_channels,out_channels,noise_emb_dim=noise_emb_dim, dropout=dropout)
            )
            current_channels= out_channels

    def forward(self, x, encoder_features,noise_emb):
        x=self.upsample(x)
        x=torch.cat([x,encoder_features], dim=1)
        for block in self.blocks:
            x=block(x,noise_emb)
        return x    

class RAD(nn.Module):
    def __init__(self,in_channels =1, out_channels=1, base_channels=128, channels_multiplier=(1,2,4),
                 nbr_rb=2, dropout=0.1, ):
        super().__init__()

        self.noise_emb_dim=base_channels
        self.spatial_noise_embedding=SpatialNoiseEmbedding(dim=base_channels)#,noise_dim=self.noise_emb_dim)

        channels=[]
        for m in channels_multiplier: channels.append(base_channels*m)
        self.in_conv= nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        current_channels= channels[0]
        encoder_channels= channels[:-1]
        bottleneck_channels=channels[-1]
        #encoder
        self.down_levels= nn.ModuleList()
        for level_out_ch in encoder_channels:
            down_level= DownLevel(current_channels, level_out_ch, nbr_rb,noise_emb_dim=self.noise_emb_dim, dropout=dropout )
            self.down_levels.append(down_level)
            current_channels= level_out_ch           
        #bottleneck
        self.bottleneck_blocks= nn.ModuleList()
        for i in range (nbr_rb):
            self.bottleneck_blocks.append(ResBlock(current_channels,bottleneck_channels, dropout=dropout,noise_emb_dim=self.noise_emb_dim))
            current_channels= bottleneck_channels
        #decoder
        self.up_levels=nn.ModuleList()
        for encoder_channels in list(reversed(encoder_channels)):
            up_level= Uplevel(current_channels, encoder_channels, encoder_channels, nbr_rb,noise_emb_dim=self.noise_emb_dim, dropout=dropout)
            self.up_levels.append(up_level)
            current_channels=encoder_channels

        self.out_normalization= RMSNorm(current_channels)
        self.activation=nn.SiLU()
        self.out_conv=nn.Conv2d(current_channels,out_channels,kernel_size=3, padding=1)


    def forward(self, xt, b_bar_t):
        
        noise_emb=self.spatial_noise_embedding(b_bar_t)
        x=self.in_conv(xt)

        encoder_features_list=[]
        for down_level in self.down_levels:
            x,encoder_features=down_level(x,noise_emb)
            encoder_features_list.append(encoder_features)
        for block in self.bottleneck_blocks:
            x=block(x,noise_emb)
        for up_level in self.up_levels:
            encoder_features=encoder_features_list.pop()
            x=up_level(x,encoder_features,noise_emb)   
        x=self.out_normalization(x)
        x=self.activation(x)
        noise_prediction= self.out_conv(x)
        
        return noise_prediction