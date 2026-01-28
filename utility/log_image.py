import torch
import wandb
import numpy as np


def log_images_wandb(model, dataset, device, indices, epoch,  threshold=0.5, alpha=0.45):
    model.eval()
    n = len(indices)
    
    wandb_images = []
    print("logging")
    with torch.no_grad():
        for r, idx in enumerate(indices):
            x, y = dataset[idx]

            # x: [C,H,W] -> [1,C,H,W]
            xb = x.unsqueeze(0).to(device)

            logits = model(xb)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            pred = (prob > threshold).astype(np.uint8)

            x2d = x[0].detach().cpu().numpy()

            y_np = y.detach().cpu().numpy()
            y2d = y_np[0] if y_np.ndim == 3 else y_np
            gt = (y2d > 0.5).astype(np.uint8)
            

            # create caption with relevant info
            caption = f"Ep: {epoch} | Pat: {idx} | Pixel GT: {gt.sum()} | Pixel Pred: {pred.sum()}"

            # save img and masks to comply with wandb format
            img = wandb.Image(
                x2d,
                masks={
                    "predictions": {"mask_data": pred, "class_labels": {0: "background", 1: "tumor"}},
                    "ground_truth": {"mask_data": gt, "class_labels": {0: "background", 1: "tumor"}}
                },
                caption=caption
            )
            wandb_images.append(img)


    
    # see all regroup in the wandb UI
    wandb.log({"image_samples": wandb_images})

