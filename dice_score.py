import torch
from torch import Tensor
from torch import sigmoid


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    target = target.float()


    probs = torch.sigmoid(input)

    if probs.dim() == 2:
        sum_dim = (-1, -2)
    elif probs.dim() == 3:
        sum_dim = (-1, -2) if reduce_batch_first else (-1, -2)
    elif probs.dim() == 4:
        sum_dim = (-1, -2) if reduce_batch_first else (-1, -2, -3)
    else:
        raise ValueError(f"Unsupported tensor shape: {probs.shape}")
    
    intersection = (probs * target).sum(dim=sum_dim)
    cardinality = probs.sum(dim=sum_dim) + target.sum(dim=sum_dim)

    dice = (2.0 * intersection + epsilon) / (cardinality + epsilon)
    dice = torch.where(cardinality == 0, torch.ones_like(dice), dice)

    return dice.mean()


def dice_eval(
    logits: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """
    Hard Dice computed ONLY on samples where GT contains tumor.
    Returns:
      dice_mean: scalar Tensor (mean over valid samples)
      n_valid:   scalar Tensor (number of valid samples used)
    Accepts shapes:
      logits:  [B,1,H,W] or [B,H,W]
      target:  [B,1,H,W] or [B,H,W] (0/1)
    """
    if logits.dim() == 3:
        logits = logits.unsqueeze(1)  
    if target.dim() == 3:
        target = target.unsqueeze(1)   

    assert logits.shape == target.shape, f"Shape mismatch: {logits.shape} vs {target.shape}"

    prob = torch.sigmoid(logits)
    pred = (prob > threshold).float()
    tgt  = (target > 0.5).float()

    dims = (1, 2, 3)  # sum over C,H,W

    # valid samples = GT has at least one positive pixel
    gt_sum = tgt.sum(dim=dims)               
    valid = gt_sum > 0                       #

    n_valid = valid.sum()

    if n_valid == 0:
        # no valid samples in this batch
        return torch.tensor(0.0, device=logits.device), torch.tensor(0, device=logits.device)

    inter = (pred * tgt).sum(dim=dims)       
    denom = pred.sum(dim=dims) + gt_sum      

    dice_per_sample = (2.0 * inter + eps) / (denom + eps)  
    dice_mean = dice_per_sample[valid].mean()

    return dice_mean, n_valid





def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
