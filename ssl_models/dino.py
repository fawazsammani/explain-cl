import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" from https://github.com/facebookresearch/dino"""

class DINOHead(nn.Module):
    
    def __init__(self, in_dim, out_dim, use_bn, norm_last_layer, nlayers, hidden_dim, bottleneck_dim):
        super().__init__()
        
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
    
class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))
    
class DINOLoss(nn.Module):
    def __init__(self, out_dim, warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs, nepochs, 
                 student_temp=0.1, center_momentum=0.9):
        super().__init__()
        
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.nepochs = nepochs
        self.teacher_temp_schedule = np.concatenate((np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
                                                     np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp))

    def forward(self, student_output, teacher_output):
        student_out = student_output / self.student_temp
        temp = self.teacher_temp_schedule[self.nepochs - 1]    # last one
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach()
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1).mean()
        return loss
    
    
class ResNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        modules = list(backbone.children())[:-2]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x).mean(dim=[2, 3])
    
class RestructuredDINO(nn.Module):
    
    def __init__(self, student, teacher):
        super().__init__()

        self.encoder_student = ResNet(student.backbone)
        self.encoder = ResNet(teacher.backbone)
        
        self.contrastive_head_student = student.head
        self.contrastive_head = teacher.head
        

    def forward(self, x, run_teacher):
        
        if run_teacher:
            x = self.encoder(x)
            x = self.contrastive_head(x)
        else:
            x = self.encoder_student(x)
            x = self.contrastive_head_student(x) 
            
        return x

    
def get_dino_model_without_loss(ckpt_path = 'dino_resnet50_pretrain_full_checkpoint.pth'):
    state_dict = torch.load('pretrained_models/dino_models/' + ckpt_path, map_location='cpu')
    state_dict_student = state_dict['student']
    state_dict_teacher = state_dict['teacher']
    
    state_dict_student = {k.replace("module.", ""): v for k, v in state_dict_student.items()}
    state_dict_teacher = {k.replace("module.", ""): v for k, v in state_dict_teacher.items()}
    
    student_backbone = torchvision.models.resnet50()
    teacher_backbone = torchvision.models.resnet50()
    embed_dim = student_backbone.fc.weight.shape[1]
    
    student_head = DINOHead(in_dim = embed_dim, out_dim = 60000, use_bn=True, norm_last_layer=True, nlayers=2, hidden_dim=4096, bottleneck_dim=256)
    teacher_head = DINOHead(in_dim = embed_dim, out_dim = 60000, use_bn =True, norm_last_layer=True, nlayers=2, hidden_dim=4096, bottleneck_dim=256)
    student_head.last_layer = nn.Linear(256, 60000, bias = False)
    teacher_head.last_layer = nn.Linear(256, 60000, bias = False)
    
    student = MultiCropWrapper(student_backbone, student_head)
    teacher = MultiCropWrapper(teacher_backbone, teacher_head)
    
    student.load_state_dict(state_dict_student)
    teacher.load_state_dict(state_dict_teacher)
    
    restructured_model = RestructuredDINO(student, teacher)
    
    return restructured_model.to(device)


def get_dino_model_with_loss(ckpt_path = 'dino_rn50_checkpoint.pth'):
    state_dict = torch.load('pretrained_models/dino_models/' + ckpt_path, map_location='cpu')
    
    state_dict_student = state_dict['student']
    state_dict_teacher = state_dict['teacher']
    state_dict_args = vars(state_dict['args'])
    state_dic_dino_loss = state_dict['dino_loss']
    
    state_dict_student = {k.replace("module.", ""): v for k, v in state_dict_student.items()}
    state_dict_teacher = {k.replace("module.", ""): v for k, v in state_dict_teacher.items()}
    
    student_backbone = torchvision.models.resnet50()
    teacher_backbone = torchvision.models.resnet50()
    embed_dim = student_backbone.fc.weight.shape[1]
    
    student_head = DINOHead(in_dim = embed_dim, 
                            out_dim = state_dict_args['out_dim'], 
                            use_bn = state_dict_args['use_bn_in_head'], 
                            norm_last_layer = state_dict_args['norm_last_layer'],    
                            nlayers = 3, 
                            hidden_dim = 2048, 
                            bottleneck_dim = 256)
    
    teacher_head = DINOHead(in_dim = embed_dim, 
                            out_dim = state_dict_args['out_dim'], 
                            use_bn = state_dict_args['use_bn_in_head'], 
                            norm_last_layer = state_dict_args['norm_last_layer'],    
                            nlayers = 3, 
                            hidden_dim = 2048, 
                            bottleneck_dim = 256)
    
    loss = DINOLoss(out_dim = state_dict_args['out_dim'], 
                    warmup_teacher_temp = state_dict_args['warmup_teacher_temp'], 
                    teacher_temp = state_dict_args['teacher_temp'], 
                    warmup_teacher_temp_epochs = state_dict_args['warmup_teacher_temp_epochs'], 
                    nepochs = state_dict_args['epochs'])
    
    student = MultiCropWrapper(student_backbone, student_head)
    teacher = MultiCropWrapper(teacher_backbone, teacher_head)
    
    student.load_state_dict(state_dict_student)
    teacher.load_state_dict(state_dict_teacher)
    loss.load_state_dict(state_dic_dino_loss)
    
    restructured_model = RestructuredDINO(student, teacher)

    return restructured_model.to(device), loss.to(device)