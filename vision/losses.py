import torch

# Continuous loss function based in Langevin dynamics
class SimpleLoss:
    def __init__(self,):
        self.mse = torch.nn.MSELoss()

    def __call__(self, net, data, label):
        x = data
        y = net(x).squeeze()
        label = label.squeeze()
        assert y.shape == label.shape, "mismatched dimensions in model output"
        loss = ((y - label) ** 2) + torch.abs(y - label) * 0.1
        # loss = self.mse(y, label)
        # loss[torch.isnan(loss)] = 0
        return loss