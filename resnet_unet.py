import torch
import torch.nn as nn
import torchvision.models as models

class ResNetUNet(nn.Module):
    def __init__(self, n_classes=40, n_channels=4):
        super().__init__()
        
        # Load pretrained ResNet
        # We use ResNet34 as it's lighter than ResNet50 but powerful enough
        self.base_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # Modify the first layer to accept n_channels (e.g. 4 for RGBD)
        # Original: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        original_conv1 = self.base_model.conv1
        self.base_model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize the new conv1 weights
        # Copy weights from the first 3 channels
        with torch.no_grad():
            self.base_model.conv1.weight[:, :3] = original_conv1.weight
            # For the 4th channel (Depth), we can initialize it with the mean of RGB weights
            # or just zero. Let's use mean to give it a good start.
            if n_channels > 3:
                self.base_model.conv1.weight[:, 3:] = torch.mean(original_conv1.weight, dim=1, keepdim=True)
        
        self.base_layers = list(self.base_model.children())
        
        # Encoder layers
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # Conv1, BN, ReLU. Size: (N, 64, H/2, W/2)
        self.maxpool = self.base_layers[3] # MaxPool. Size: (N, 64, H/4, W/4)
        self.layer1 = nn.Sequential(*self.base_layers[4]) # ResBlock 1. Size: (N, 64, H/4, W/4)
        self.layer2 = nn.Sequential(*self.base_layers[5]) # ResBlock 2. Size: (N, 128, H/8, W/8)
        self.layer3 = nn.Sequential(*self.base_layers[6]) # ResBlock 3. Size: (N, 256, H/16, W/16)
        self.layer4 = nn.Sequential(*self.base_layers[7]) # ResBlock 4. Size: (N, 512, H/32, W/32)
        
        # Decoder layers
        # up4 takes layer4 (512) and layer3 (256). Out: 256
        self.up4 = self.up_block(512 + 256, 256)
        # up3 takes up4_out (256) and layer2 (128). Out: 128
        self.up3 = self.up_block(256 + 128, 128)
        # up2 takes up3_out (128) and layer1 (64). Out: 64
        self.up2 = self.up_block(128 + 64, 64)
        # up1 takes up2_out (64) and layer0 (64). Out: 64
        self.up1 = self.up_block(64 + 64, 64)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1)
        )
        
        self.n_classes = n_classes

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x0 = self.layer0(x) # H/2
        x_mp = self.maxpool(x0) # H/4
        x1 = self.layer1(x_mp) # H/4
        x2 = self.layer2(x1) # H/8
        x3 = self.layer3(x2) # H/16
        x4 = self.layer4(x3) # H/32
        
        # Decoder
        
        # Block 4: x4 (H/32) -> H/16. Concat with x3.
        x = torch.nn.functional.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
        if x.size() != x3.size():
            x = torch.nn.functional.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x3], dim=1)
        x = self.up4(x)
        
        # Block 3: x (H/16) -> H/8. Concat with x2.
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if x.size() != x2.size():
            x = torch.nn.functional.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x2], dim=1)
        x = self.up3(x)
        
        # Block 2: x (H/8) -> H/4. Concat with x1.
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if x.size() != x1.size():
            x = torch.nn.functional.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)
        x = self.up2(x)
        
        # Block 1: x (H/4) -> H/2. Concat with x0.
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if x.size() != x0.size():
            x = torch.nn.functional.interpolate(x, size=x0.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x0], dim=1)
        x = self.up1(x)
        
        # Final upsample: H/2 -> H
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        logits = self.final_conv(x)
        
        return logits

if __name__ == '__main__':
    model = ResNetUNet(n_classes=41, n_channels=4)
    x = torch.randn(1, 4, 480, 640)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
