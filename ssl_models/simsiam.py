import torch
import torch.nn as nn
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""from https://github.com/facebookresearch/simsiam"""

class SimSiam(nn.Module):

    def __init__(self, base_encoder, dim, pred_dim):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        symetric is True only when training
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        z1 = self.encoder(x1).detach() # NxC
        z2 = self.encoder(x2).detach() # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC
        
        loss = -(nn.CosineSimilarity(dim=1)(p1, z2).mean() + nn.CosineSimilarity(dim=1)(p2, z1).mean()) * 0.5
        
        return loss
    
class ResNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        modules = list(backbone.children())[:-2]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x).mean(dim=[2, 3])
    
class RestructuredSimSiam(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.encoder = ResNet(model.encoder)
        self.mlp_encoder = model.encoder.fc
        self.mlp_encoder[6].bias.requires_grad = False
        self.contrastive_head = model.predictor

    def forward(self, x, run_head = True):
        
        x = self.mlp_encoder(self.encoder(x))   # don't detach since we will do backprop for explainability
        
        if run_head:
            x = self.contrastive_head(x) 
            
        return x

    
def get_simsiam(ckpt_path = 'checkpoint_0099.pth.tar'):

    model = SimSiam(base_encoder = torchvision.models.resnet50, 
                    dim = 2048, 
                    pred_dim = 512)
    
    checkpoint = torch.load('pretrained_models/simsiam_models/'+ ckpt_path, map_location='cpu')
    state_dic = checkpoint['state_dict']
    state_dic = {k.replace("module.", ""): v for k, v in state_dic.items()}
    model.load_state_dict(state_dic)
    restructured_model = RestructuredSimSiam(model)
    return restructured_model.to(device)
