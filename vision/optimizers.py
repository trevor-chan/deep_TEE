import torch

class AnnealedAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, t_up = 1<<14, t_down = 1<<20):
        self.base_lr = lr
        self.t_up = t_up
        self.t_down = t_down
        self.optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.schedule)
        self.step()
        self.step_lr()
        
    def schedule(self, count):
        # rampup = 1 - 1 / (count / torch.sqrt(torch.tensor(self.t_up)) + 1)
        rampup = count / torch.tensor(self.t_up) if count < self.t_up else 1
        decay = torch.min(torch.ones_like(torch.tensor(count)), 1 / torch.sqrt(count / torch.tensor(self.t_down)))
        factor = rampup * decay
        return factor

    @torch.compile
    def step(self):
        self.optimizer.step()
        
    def step_lr(self):
        self.scheduler.step()
        
    def __getattr__(self, name):
        return getattr(self.optimizer, name)
    
    def get_lr(self):
        return self.scheduler.get_last_lr()[0].item()

    def zero_grad(self, *args, **kwargs):
        self.optimizer.zero_grad(*args, **kwargs)