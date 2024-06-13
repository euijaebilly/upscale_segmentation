import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.enc5 = CBR(512, 1024)

        self.pool = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.dec1 = CBR(1024, 512)
        self.dec2 = CBR(512, 256)
        self.dec3 = CBR(256, 128)
        self.dec4 = CBR(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        dec1 = self.up1(enc5)
        dec1 = torch.cat((dec1, enc4[:, :, :dec1.shape[2], :dec1.shape[3]]), dim=1)
        dec1 = self.dec1(dec1)

        dec2 = self.up2(dec1)
        dec2 = torch.cat((dec2, enc3[:, :, :dec2.shape[2], :dec2.shape[3]]), dim=1)
        dec2 = self.dec2(dec2)

        dec3 = self.up3(dec2)
        dec3 = torch.cat((dec3, enc2[:, :, :dec3.shape[2], :dec3.shape[3]]), dim=1)
        dec3 = self.dec3(dec3)

        dec4 = self.up4(dec3)
        dec4 = torch.cat((dec4, enc1[:, :, :dec4.shape[2], :dec4.shape[3]]), dim=1)
        dec4 = self.dec4(dec4)

        return self.final(dec4)
