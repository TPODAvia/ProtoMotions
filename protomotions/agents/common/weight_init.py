import torch.nn as nn

def weight_init(m, orthogonal=False):
    if isinstance(m, nn.Linear):
        if orthogonal:
            nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif hasattr(m, "reset_parameters"):
        m.reset_parameters() 