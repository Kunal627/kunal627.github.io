import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the U-Net model

class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(Unet, self).__init__()

        self.enc1 = self.conv2dblk(in_channels, 64)
        self.enc2 = self.conv2dblk(64, 128)
        self.enc3 = self.conv2dblk(128, 256)
        self.enc4 = self.conv2dblk(256, 512)

        self.bottleneck = self.conv2dblk(512, 1024)

        self.upconv4 = self.upconv(1024, 512)
        self.dec4 = self.conv2dblk(1024, 512)

        self.upconv3 = self.upconv(512, 256)
        self.dec3 = self.conv2dblk(512, 256)

        self.upconv2 = self.upconv(256, 128)
        self.dec2 = self.conv2dblk(256, 128)

        self.upconv1 = self.upconv(128, 64)
        self.dec1 = self.conv2dblk(128, 64)

        self.final_layer = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv2dblk(self, in_ch, out_ch, kersz=3):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_ch,  out_channels=out_ch, kernel_size=kersz, stride=1, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch,  out_channels=out_ch, kernel_size=kersz, stride=1, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True)
        )
    
    def upconv(self, in_ch, out_ch, kersz=2):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kersz, stride=2),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):

        # downsample 
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, kernel_size=2))
        e3 = self.enc3(F.max_pool2d(e2, kernel_size=2))
        e4 = self.enc4(F.max_pool2d(e3, kernel_size=2))

        bottleneck = self.bottleneck(F.max_pool2d(e4, kernel_size=2))

        # upsample
        up4 = self.upconv4(bottleneck)
        up4 = torch.cat((e4, up4), dim=1)
        d4 = self.dec4(up4)

        up3 = self.upconv3(d4)
        up3 = torch.cat((e3, up3), dim=1)
        d3 = self.dec3(up3)

        up2 = self.upconv2(d3)
        up2 = torch.cat((e2, up2), dim=1)
        d2 = self.dec2(up2)

        up1 = self.upconv1(d2)
        up1 = torch.cat((e1, up1), dim=1)
        d1 = self.dec1(up1)

        out = self.final_layer(d1)
        
        return out
    

#model = Unet(in_channels=3, out_channels=3)  # For RGB images
#x = torch.randn(1, 3, 32, 32)  # Example input tensor (batch size, channels, height, width)
#output = model(x)  # Example output tensor
#print("Output shape:", output.shape)
#torch.save(model.state_dict(), f"./out/checkpoints/diffusion/unet_diff_model_epoch.pt")



