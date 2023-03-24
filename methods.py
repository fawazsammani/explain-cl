import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from PIL import Image
from sklearn.decomposition import NMF
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def relu_hook_function(module, grad_in, grad_out):
    if isinstance(module, nn.ReLU):
        return (F.relu(grad_in[0]),)
    
def blur_sailency(input_image):
    return torchvision.transforms.functional.gaussian_blur(input_image, kernel_size=[11, 11], sigma=[5,5])

def occlusion(img1, img2, model, w_size = 64, stride = 8, batch_size = 32):
    
    measure = nn.CosineSimilarity(dim=-1) 
    output_size = int(((img2.size(-1) - w_size) / stride) + 1)
    out1_condition, out2_condition = model(img1), model(img2)
    images1 = []
    images2 = []

    for i in range(output_size):
        for j in range(output_size):
            start_i, start_j = i * stride, j * stride
            image1 = img1.clone().detach()
            image2 = img2.clone().detach()
            image1[:, :, start_i : start_i + w_size, start_j : start_j + w_size] = 0  
            image2[:, :, start_i : start_i + w_size, start_j : start_j + w_size] = 0  
            images1.append(image1)
            images2.append(image2)

    images1 = torch.cat(images1, dim=0).to(device)
    images2 = torch.cat(images2, dim=0).to(device)

    score_map1 = []
    score_map2 = []

    assert images1.shape[0] == images2.shape[0]

    for b in range(0, images2.shape[0], batch_size):

        with torch.no_grad():
            out1 = model(images1[b : b + batch_size, :])
            out2 = model(images2[b : b + batch_size, :])

        score_map1.append(measure(out1, out2_condition))  # try torch.mm(out2_condition, out1.t())[0]
        score_map2.append(measure(out1_condition, out2))  # try torch.mm(out1_condition, out2.t())[0]

    score_map1 = torch.cat(score_map1, dim = 0)   
    score_map2 = torch.cat(score_map2, dim = 0)    
    assert images2.shape[0] == score_map2.shape[0] == score_map1.shape[0]

    heatmap1 = score_map1.view(output_size, output_size).cpu().detach().numpy()
    heatmap2 = score_map2.view(output_size, output_size).cpu().detach().numpy()
    base_score = measure(out1_condition, out2_condition)

    heatmap1 = (heatmap1 - base_score.item()) * -1   # or base_score.item() - heatmap1. The higher the drop, the better
    heatmap2 = (heatmap2 - base_score.item()) * -1   # or base_score.item() - heatmap2. The higher the drop, the better
    
    heatmap1 = (heatmap1 - heatmap1.min()) / (heatmap1.max() - heatmap1.min())
    heatmap2 = (heatmap2 - heatmap2.min()) / (heatmap2.max() - heatmap2.min())
    
    return heatmap1, heatmap2

def pairwise_occlusion(img1, img2, model, batch_size, erase_scale, erase_ratio, num_erases):

    measure = nn.CosineSimilarity(dim=-1) 
    out1_condition, out2_condition = model(img1), model(img2)
    baseline = measure(out1_condition, out2_condition).detach()
    # a bit sensitive to scale and ratio. erase_scale is from (scale[0] * 100) % to (scale[1] * 100) %
    random_erase = transforms.RandomErasing(p=1.0, scale=erase_scale, ratio=erase_ratio)  
    
    image1 = img1.clone().detach()
    image2 = img2.clone().detach()
    images1 = []
    images2 = []

    for _ in range(num_erases):
        images1.append(random_erase(image1))
        images2.append(random_erase(image2))

    images1 = torch.cat(images1, dim=0).to(device)
    images2 = torch.cat(images2, dim=0).to(device)
    
    sims = []
    weights1 = []
    weights2 = []

    for b in range(0, images2.shape[0], batch_size):

        with torch.no_grad():
            out1 = model(images1[b : b + batch_size, :])
            out2 = model(images2[b : b + batch_size, :])
            sims.append(measure(out1, out2))
            weights1.append(out1.norm(dim=-1))
            weights2.append(out2.norm(dim=-1))

    sims = torch.cat(sims, dim = 0)       
    weights1, weights2 = torch.cat(weights1, dim = 0).cpu().numpy(), torch.cat(weights2, dim = 0).cpu().numpy()
    weights = list(zip(weights1, weights2))
    sims = baseline - sims   # the higher the drop, the better
    sims = F.softmax(sims, dim = -1)
    sims = sims.cpu().numpy()

    assert sims.shape[0] == images1.shape[0] == images2.shape[0]
    A1 = np.zeros((224, 224))
    A2 = np.zeros((224, 224))

    for n in range(images1.shape[0]):

        im1_2d = images1[n].cpu().numpy().transpose((1, 2, 0)).sum(axis=-1)
        im2_2d = images2[n].cpu().numpy().transpose((1, 2, 0)).sum(axis=-1)

        joint_similarity = sims[n]
        weight = weights[n]

        if weight[0] < weight[1]:
            A1[im1_2d == 0] += joint_similarity
        else:
            A2[im2_2d == 0] += joint_similarity

    A1 = A1 / (np.max(A1) + 1e-9)  
    A2 = A2 / (np.max(A2) + 1e-9)

    return A1, A2

def create_mixed_images(transform_type, ig_transforms, step, img_path, add_noise):

    img = Image.open(img_path).convert('RGB') if isinstance(img_path, str) else img_path
    img1 = ig_transforms['pure'](img).unsqueeze(0).to(device)
    img2 = ig_transforms[transform_type](img).unsqueeze(0).to(device)

    lambdas = np.arange(1,0,-step)
    mixed_images = []
    for l,lam in enumerate(lambdas):
        mixed_img = lam * img1 + (1 - lam) * img2
        mixed_images.append(mixed_img)
        
    if add_noise:
        sigma = 0.15 / (torch.max(img1) - torch.min(img1)).item()
        mixed_images = [im + torch.zeros_like(im).normal_(0, sigma) if (n>0) and (n<len(mixed_images)-1) else im for n,im in enumerate(mixed_images)]
        
    return mixed_images

def averaged_transforms(guided, ssl_model, mixed_images, blur_output):

    measure = nn.CosineSimilarity(dim=-1)

    if guided:
        handles = []
        for i, module in enumerate(ssl_model.modules()):
            if isinstance(module, nn.ReLU):
                handles.append(module.register_backward_hook(relu_hook_function))
                
    grads1 = []
    grads2 = []

    for xbar_image in mixed_images[1:]:  
        
        ssl_model.zero_grad()
        input_image1 = mixed_images[0].clone().requires_grad_()
        input_image2 = xbar_image.clone().requires_grad_()

        if input_image1.grad is not None:
            input_image1.grad.data.zero_()
            input_image2.grad.data.zero_()

        score = measure(ssl_model(input_image1), ssl_model(input_image2))
        score.backward()
        grads1.append(input_image1.grad.data)
        grads2.append(input_image2.grad.data)

    grads1 = torch.cat(grads1).mean(0).unsqueeze(0)
    grads2 = torch.cat(grads2).mean(0).unsqueeze(0)

    sailency1, _ = torch.max((mixed_images[0] * grads1).abs(), dim=1)
    sailency2, _ = torch.max((mixed_images[-1] * grads2).abs(), dim=1)

    if guided:     # remove handles after finishing
        for handle in handles:
            handle.remove()
            
    if blur_output:
        sailency1 = blur_sailency(sailency1)
        sailency2 = blur_sailency(sailency2)
            
    return sailency1, sailency2

def sailency(guided, ssl_model, img1, img2, blur_output):
    
    measure = nn.CosineSimilarity(dim=-1)
    
    if guided:
        handles = []
        for i, module in enumerate(ssl_model.modules()):
            if isinstance(module, nn.ReLU):
                handles.append(module.register_backward_hook(relu_hook_function))
                
    input_image1 = img1.clone().requires_grad_()
    input_image2 = img2.clone().requires_grad_()
    score = measure(ssl_model(input_image1), ssl_model(input_image2))
    score.backward()
    grads1 = input_image1.grad.data
    grads2 = input_image2.grad.data   
    sailency1, _ = torch.max((img1 * grads1).abs(), dim=1)
    sailency2, _ = torch.max((img2 * grads2).abs(), dim=1)

    if guided:     # remove handles after finishing
        for handle in handles:
            handle.remove()
            
    if blur_output:
        sailency1 = blur_sailency(sailency1)
        sailency2 = blur_sailency(sailency2)
            
    return sailency1, sailency2

def smooth_grad(guided, ssl_model, img1, img2, blur_output, steps = 50):
    
    measure = nn.CosineSimilarity(dim=-1)
    sigma = 0.15 / (torch.max(img1) - torch.min(img1)).item()
    
    if guided:
        handles = []
        for i, module in enumerate(ssl_model.modules()):
            if isinstance(module, nn.ReLU):
                handles.append(module.register_backward_hook(relu_hook_function))
                
    noise_images1 = []
    noise_images2 = []
    
    for _ in range(steps):
        noise = torch.zeros_like(img1).normal_(0, sigma)
        noise_images1.append(img1 + noise)   
        noise_images2.append(img2 + noise)
                
    grads1 = []
    grads2 = []

    for n1, n2 in zip(noise_images1, noise_images2):  
        input_image1 = n1.clone().requires_grad_()
        input_image2 = n2.clone().requires_grad_()

        if input_image1.grad is not None:
            input_image1.grad.data.zero_()
            input_image2.grad.data.zero_()

        score = measure(ssl_model(input_image1), ssl_model(input_image2))
        score.backward()
        grads1.append(input_image1.grad.data)
        grads2.append(input_image2.grad.data)

    grads1 = torch.cat(grads1).mean(0).unsqueeze(0)
    grads2 = torch.cat(grads2).mean(0).unsqueeze(0)   
    sailency1, _ = torch.max((img1 * grads1 ).abs(), dim=1)
    sailency2, _ = torch.max((img2 * grads2).abs(), dim=1)

    if guided:     # remove handles after finishing
        for handle in handles:
            handle.remove()
            
    if blur_output:
        sailency1 = blur_sailency(sailency1)
        sailency2 = blur_sailency(sailency2)
            
    return sailency1, sailency2

class GradCAM(nn.Module):
    
    def __init__(self, ssl_model):
        super(GradCAM, self).__init__()
        
        self.gradients = {}
        self.features = {}
        
        self.feature_extractor = ssl_model.encoder.net
        self.contrastive_head = ssl_model.contrastive_head
        self.measure = nn.CosineSimilarity(dim=-1)
        
    def save_grads(self, img_index):
    
        def hook(grad):
            self.gradients[img_index] = grad.detach()

        return hook
    
    def save_features(self, img_index, feats):
        self.features[img_index] = feats.detach()
    
    def forward(self, img1, img2):
        
        features1 = self.feature_extractor(img1)
        features2 = self.feature_extractor(img2)
        
        self.save_features('1', features1)
        self.save_features('2', features2)
        
        h1 = features1.register_hook(self.save_grads('1'))
        h2 = features2.register_hook(self.save_grads('2'))
        
        out1, out2 = features1.mean(dim=[2, 3]), features2.mean(dim=[2, 3])
        out1, out2 = self.contrastive_head(out1), self.contrastive_head(out2)
        score = self.measure(out1, out2)
        
        return score
    
def weight_activation(feats, grads):
    cam =  feats * F.relu(grads)
    cam = torch.sum(cam, dim=1).squeeze().cpu().detach().numpy()
    return cam

def get_gradcam(ssl_model, img1, img2):
    
    grad_cam = GradCAM(ssl_model).to(device)
    score = grad_cam(img1, img2)
    grad_cam.zero_grad()
    score.backward()

    cam1 = weight_activation(grad_cam.features['1'], grad_cam.gradients['1'])
    cam2 = weight_activation(grad_cam.features['2'], grad_cam.gradients['2'])
    return cam1, cam2

def get_interactioncam(ssl_model, img1, img2, reduction, grad_interact = False):
    
    grad_cam = GradCAM(ssl_model).to(device)
    score = grad_cam(img1, img2)
    grad_cam.zero_grad()
    score.backward()
    
    G1 = grad_cam.gradients['1']
    G2 = grad_cam.gradients['2']
    
    if grad_interact:
        B, D, H, W = G1.size()
        G1_ = G1.permute(0,2,3,1).view(B, H * W, D)
        G2_ = G2.permute(0,2,3,1).view(B, H * W, D)
        G_ = torch.bmm(G1_.permute(0,2,1), G2_)  # (B, D, D)
        G1, _ = torch.max(G_, dim = -1)   # (B, D)
        G2, _ = torch.max(G_, dim = 1)    # (B, D)
        G1 = G1.unsqueeze(-1).unsqueeze(-1)
        G2 = G2.unsqueeze(-1).unsqueeze(-1)

    if reduction == 'mean':
        joint_weight = grad_cam.features['1'].mean([2,3]) * grad_cam.features['2'].mean([2,3])
    elif reduction == 'max':
        max_pooled1 = F.max_pool2d(grad_cam.features['1'], kernel_size=grad_cam.features['1'].size()[2:]).squeeze(-1).squeeze(-1)
        max_pooled2 = F.max_pool2d(grad_cam.features['2'], kernel_size=grad_cam.features['2'].size()[2:]).squeeze(-1).squeeze(-1)
        joint_weight = max_pooled1 * max_pooled2
    else:
        B, D, H, W = grad_cam.features['1'].size()
        reshaped1 = grad_cam.features['1'].permute(0,2,3,1).reshape(B, H * W, D)
        reshaped2 = grad_cam.features['2'].permute(0,2,3,1).reshape(B, H * W, D)
        features1_query, features2_query = reshaped1.mean(1).unsqueeze(1), reshaped2.mean(1).unsqueeze(1)
        attn1 = (features1_query @ reshaped1.transpose(-2, -1)).softmax(dim=-1)
        attn2 = (features2_query @ reshaped2.transpose(-2, -1)).softmax(dim=-1)
        att_reduced1 = (attn1 @ reshaped1).squeeze(1)
        att_reduced2 = (attn2 @ reshaped2).squeeze(1)
        joint_weight = att_reduced1 * att_reduced2
        
    joint_weight = joint_weight.unsqueeze(-1).unsqueeze(-1).expand_as(grad_cam.features['1'])
    
    feats1 = grad_cam.features['1'] * joint_weight
    feats2 = grad_cam.features['2'] * joint_weight

    cam1 = weight_activation(feats1, G1)
    cam2 = weight_activation(feats2, G2)
    
    return cam1, cam2
