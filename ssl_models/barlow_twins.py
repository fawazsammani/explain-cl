import torch
import torch.nn as nn
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""from https://github.com/facebookresearch/barlowtwins"""

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwins(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [2048] + list(map(int, '8192-8192-8192'.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + 0.0051 * off_diag
        return loss
    
class ResNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        modules = list(backbone.children())[:-2]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x).mean(dim=[2, 3])
    
class RestructuredBarlowTwins(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.encoder = ResNet(model.backbone)
        self.contrastive_head = model.projector

    def forward(self, x):
        x = self.encoder(x)
        x = self.contrastive_head(x)
        return x

    
def get_barlow_twins_model(ckpt_path = 'barlow_twins.pth'):
    model = BarlowTwins()
    state_dict = torch.load('pretrained_models/barlow_models/' + ckpt_path, map_location='cpu')
    state_dict = state_dict['model']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    restructured_model = RestructuredBarlowTwins(model)
    return restructured_model.to(device)