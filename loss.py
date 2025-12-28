import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W) - logits
        # targets: (B, H, W) - class indices
        
        num_classes = inputs.shape[1]
        
        # Apply Softmax to get probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Create mask to ignore specific index (e.g., background)
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float().unsqueeze(1)
            # Apply mask to both inputs and targets
            # Note: This is a simplified way. For strict Dice on valid pixels:
            # We should only compute Dice for classes that are NOT ignore_index
            # Or zero out the contributions of ignore_index
            
            # Better approach for multiclass Dice with ignore_index:
            # Calculate Dice for each class separately, then average, skipping ignore_index
            
            dice_loss = 0.0
            count = 0
            
            for c in range(num_classes):
                if c == self.ignore_index:
                    continue
                
                input_flat = inputs[:, c].contiguous().view(-1)
                target_flat = targets_one_hot[:, c].contiguous().view(-1)
                
                intersection = (input_flat * target_flat).sum()
                
                dice = (2. * intersection + self.smooth) / (input_flat.sum() + target_flat.sum() + self.smooth)
                dice_loss += 1 - dice
                count += 1
            
            if count > 0:
                return dice_loss / count
            else:
                return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        else:
            # Standard Multiclass Dice Loss
            intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
            union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            return 1 - dice.mean()

class DiceCELoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, lambda_dice=0.5, lambda_ce=0.5, label_smoothing=0.0):
        super(DiceCELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, label_smoothing=label_smoothing)
        # 如果 ignore_index 是 PyTorch 默认的 -100，DiceLoss 不需要特殊处理（除非你有类索引是 -100）
        # 如果想让 DiceLoss 计算所有类别（包括背景），可以将 ignore_index 设为 None
        dice_ignore = ignore_index if ignore_index >= 0 else None
        self.dice = DiceLoss(ignore_index=dice_ignore)
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.lambda_ce * ce_loss + self.lambda_dice * dice_loss
