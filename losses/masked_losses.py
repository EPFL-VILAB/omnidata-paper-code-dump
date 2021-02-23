import torch

def masked_l1_loss(preds, target, mask_valid):
    element_wise_loss = abs(preds - target)
    element_wise_loss[~mask_valid] = 0
    return element_wise_loss.sum() / mask_valid.sum()

def masked_mse_loss(preds, target, mask_valid):
    element_wise_loss = (preds - target)**2
    element_wise_loss[~mask_valid] = 0
    return element_wise_loss.sum() / mask_valid.sum()

def masked_loss(element_wise_loss, mask_valid):
    element_wise_loss[~mask_valid] = 0
    if mask_valid.sum() == 0:
        return torch.tensor(0.0).to(element_wise_loss.device)
    return element_wise_loss.sum() / mask_valid.sum()