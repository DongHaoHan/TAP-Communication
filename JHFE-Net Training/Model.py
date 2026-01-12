import torch
import torch.nn as nn


# ==========================================
# 1. Attention Module
# ==========================================
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.SiLU = nn.SiLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.SiLU(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ==========================================
# 2. Main Network Architecture (JHFE-Net)
# ==========================================
class JHFE_Net(nn.Module):
    def __init__(self, h_classes=6, f_classes=3, grid_size=41):
        super(JHFE_Net, self).__init__()

        self.height_emb = nn.Embedding(h_classes, grid_size * grid_size)
        self.freq_emb = nn.Embedding(f_classes, grid_size * grid_size)
        self.act = nn.SiLU(inplace=True)
        self.grid_size = grid_size

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), self.act,
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), self.act
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), self.act,
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), self.act
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), self.act,
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), self.act
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), self.act,
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), self.act
        )

        self.att1 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.att3 = AttentionBlock(F_g=32, F_l=32, F_int=16)

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(128), self.act
        )
        self.decoder1_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), self.act
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64), self.act
        )
        self.decoder2_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), self.act
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32), self.act
        )
        self.decoder3_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), self.act
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(32), self.act
        )
        self.decoder4_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), self.act
        )

        self.final_layer = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), self.act,
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            self.act
        )

    # ==========================================
    # 3. Forward Propagation
    # ==========================================
    def forward(self, img, label_height, label_freq):
        feature_height = self.height_emb(label_height).view(-1, 1, self.grid_size, self.grid_size)
        feature_freq = self.freq_emb(label_freq).view(-1, 1, self.grid_size, self.grid_size)

        x = torch.cat([img, feature_height, feature_freq], dim=1)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        dec1 = self.decoder1(enc4)
        att1 = self.att1(g=dec1, x=enc3)
        dec1 = torch.cat([dec1, att1], dim=1)
        dec1 = self.decoder1_conv(dec1)

        dec2 = self.decoder2(dec1)
        att2 = self.att2(g=dec2, x=enc2)
        dec2 = torch.cat([dec2, att2], dim=1)
        dec2 = self.decoder2_conv(dec2)

        dec3 = self.decoder3(dec2)
        att3 = self.att3(g=dec3, x=enc1)
        dec3 = torch.cat([dec3, att3], dim=1)
        dec3 = self.decoder3_conv(dec3)

        dec4 = self.decoder4(dec3)
        dec4 = self.decoder4_conv(dec4)

        out = self.final_layer(dec4)
        return out