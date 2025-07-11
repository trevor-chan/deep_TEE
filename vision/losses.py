import torch


class SimpleLoss:
    def __init__(self,):
        self.mse = torch.nn.MSELoss()

    def __call__(self, net, data, label):
        x = data
        y = net(x).squeeze()
        label = label.squeeze()
        assert y.shape == label.shape, "mismatched dimensions in model output"
        # loss = ((y - label) ** 2) + torch.abs(y - label) * 0.1
        loss = self.mse(y, label)
        # loss[torch.isnan(loss)] = 0
        return loss

class MaskedLoss:
    # Use when the label is expected to contain nan values - will use a masked_tensor to exclude nans from gradient computation
    def __init__(self,):
        self.mse = torch.nn.MSELoss()

    def __call__(self, net, data, label):
        x = data
        y = net(x).squeeze()
        label = label.squeeze()
        assert y.shape == label.shape, f"mismatched dimensions in model output, {y.shape} != {label.shape}"
        loss = ((y - label) ** 2)
        
        mask = ~torch.isnan(loss)
        masked_loss = loss * mask
        masked_loss = masked_loss / mask.sum()
        masked_loss = torch.nan_to_num(masked_loss, nan=0.0)
        return masked_loss


        # mask = ~torch.isnan(loss)
        # mx = torch.masked.masked_tensor(loss, mask, requires_grad=True)
        # my = torch.masked.masked_tensor(torch.ones_like(loss), ~mask, requires_grad=True)
        # masked_loss = torch.where(mask, mx, my)
        # return masked_loss