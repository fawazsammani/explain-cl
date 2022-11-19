import torch
import torch.nn as nn
import torchvision
import torchvision.transforms
import numpy as np
import math
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def interpolate_map(heatmap, size = (224, 224)):
    "used to interpolate heatmap to use for evaluation metrics"
    
    if not torch.is_tensor(heatmap):
        heatmap = torch.from_numpy(heatmap)
        
    heatmap = heatmap.unsqueeze(0)
    heatmap = torchvision.transforms.Resize(size)(heatmap)
    return heatmap

def blur_image(input_image):
    return torchvision.transforms.functional.gaussian_blur(input_image, kernel_size=[11, 11], sigma=[5,5])

def insertion_deletion_simult_single(model, img1, img2, explanation1, explanation2, mode, step = 224):
    
    measure = nn.CosineSimilarity(dim=-1)   
    H,W = img1.shape[-1] , img1.shape[-1]
    HW = H * W
    n_steps = (HW + step - 1) // step

    if mode == 'del':
        start1 = img1.clone()
        start2 = img2.clone()
        finish1 = torch.zeros_like(img1).reshape(1, 3, HW)
        finish2 = torch.zeros_like(img2).reshape(1, 3, HW)
    else:  # insertion
        start1 = blur_image(img1)
        start2 = blur_image(img2)
        finish1 = img1.clone().reshape(1, 3, HW)
        finish2 = img2.clone().reshape(1, 3, HW)

    scores = np.empty(n_steps + 1)
    salient_order1 = torch.flip(explanation1.view(-1, HW).argsort(), dims=[-1])
    salient_order2 = torch.flip(explanation2.view(-1, HW).argsort(), dims=[-1])
    
    for i in range(n_steps+1):
        
        score = measure(model(start1.to(device)), model(start2.to(device)))
        scores[i] = score[0].item()

        if i < n_steps:
            coords1 = salient_order1[:, step * i : step * (i + 1)]
            coords2 = salient_order2[:, step * i : step * (i + 1)]
            start1, start2 = start1.reshape(1, 3, HW), start2.reshape(1, 3, HW)
            start1[:, :, coords1[0]] = finish1[:, :, coords1[0]]
            start2[:, :, coords2[0]] = finish2[:, :, coords2[0]]
            start1, start2 = start1.reshape(1, 3, H, W), start2.reshape(1, 3, H, W)
            
    return (scores.sum() - scores[0] / 2 - scores[-1] / 2) / (scores.shape[0] - 1)

def insertion_deletion_simult_batch(model, img1_batch, img2_batch, explanation1_batch, explanation2_batch, batch_size, mode, step = 224):
    
    measure = nn.CosineSimilarity(dim=-1)   
    H,W = img1_batch.shape[-1] , img1_batch.shape[-1]
    HW = H * W
    n_samples = img1_batch.shape[0]
    assert n_samples % batch_size == 0
    
    n_steps = (HW + step - 1) // step

    substrate1 = torch.zeros_like(img1_batch)
    substrate2 = torch.zeros_like(img2_batch)
    
    if mode == 'del':
        start1 = img1_batch.clone()
        start2 = img2_batch.clone()
        finish1 = substrate1
        finish2 = substrate2
    else:
        for j in range(n_samples // batch_size):
            substrate1[j * batch_size: (j+1) * batch_size] = blur_image(img1_batch[j * batch_size: (j+1) * batch_size])
            substrate2[j * batch_size: (j+1) * batch_size] = blur_image(img2_batch[j * batch_size: (j+1) * batch_size])
            
        start1 = substrate1
        start2 = substrate2
        finish1 = img1_batch.clone()
        finish2 = img2_batch.clone()
        
    scores = torch.zeros(n_steps + 1, n_samples)
    salient_order1 = torch.flip(explanation1_batch.view(-1, HW).argsort(), dims=[-1])
    salient_order2 = torch.flip(explanation2_batch.view(-1, HW).argsort(), dims=[-1])

    for i in range(n_steps+1):
        
        for j in range(n_samples // batch_size):  # Iterate over batches
            
            with torch.no_grad():
                out_mini_batch1 = model(start1[j * batch_size: (j+1) * batch_size].to(device))
                out_mini_batch2 = model(start2[j * batch_size: (j+1) * batch_size].to(device))
                scores[i, j * batch_size: (j+1) * batch_size] = measure(out_mini_batch1, out_mini_batch2)
            
        coords1 = salient_order1[:, step * i : step * (i + 1)]
        coords2 = salient_order2[:, step * i : step * (i + 1)]
        
        start1, start2 = start1.reshape(n_samples, 3, HW), start2.reshape(n_samples, 3, HW)
        finish1, finish2 = finish1.reshape(n_samples, 3, HW), finish2.reshape(n_samples, 3, HW)
        
        for n in range(n_samples):
            start1[n, :, coords1[n]] = finish1[n, :, coords1[n]]
            start2[n, :, coords2[n]] = finish2[n, :, coords2[n]]
        
        start1, start2 = start1.reshape(n_samples, 3, H, W), start2.reshape(n_samples, 3, H, W)
        finish1, finish2 = finish1.reshape(n_samples, 3, H, W), finish2.reshape(n_samples, 3, H, W)
        
    scores = scores.mean(1)
    return (scores.sum() - scores[0] / 2 - scores[-1] / 2) / (scores.shape[0] - 1)

def insertion_deletion_cond_single(model, img, cond_img, explanation, mode, step = 224):
    
    measure = nn.CosineSimilarity(dim=-1)   
    H,W = img.shape[-1] , img.shape[-1]
    HW = H * W
    n_steps = (HW + step - 1) // step

    if mode == 'del':
        start = img.clone()
        finish = torch.zeros_like(img).reshape(1, 3, HW)
    else:  # insertion
        start = blur_image(img)
        finish = img.clone().reshape(1, 3, HW)

    scores = np.empty(n_steps + 1)
    salient_order = torch.flip(explanation.view(-1, HW).argsort(), dims=[-1])
    
    for i in range(n_steps+1):
        
        score = measure(model(start.to(device)), model(cond_img.to(device)))
        scores[i] = score[0].item()

        if i < n_steps:
            coords = salient_order[:, step * i : step * (i + 1)]
            start = start.reshape(1, 3, HW)
            start[:, :, coords[0]] = finish[:, :, coords[0]]
            start = start.reshape(1, 3, H, W)
            
    return (scores.sum() - scores[0] / 2 - scores[-1] / 2) / (scores.shape[0] - 1)

def insertion_deletion_cond_batch(model, img_batch, cond_img_batch, explanation_batch, batch_size, mode, step = 224):
    
    measure = nn.CosineSimilarity(dim=-1)   
    H,W = img_batch.shape[-1] , img_batch.shape[-1]
    HW = H * W
    n_samples = img_batch.shape[0]
    assert n_samples % batch_size == 0
    
    n_steps = (HW + step - 1) // step

    substrate = torch.zeros_like(img_batch)

    if mode == 'del':
        start = img_batch.clone()
        finish = substrate
    else:
        for j in range(n_samples // batch_size):
            substrate[j * batch_size: (j+1) * batch_size] = blur_image(img_batch[j * batch_size: (j+1) * batch_size])
            
        start = substrate
        finish = img_batch.clone()
        
    scores = torch.zeros(n_steps + 1, n_samples)
    salient_order = torch.flip(explanation_batch.view(-1, HW).argsort(), dims=[-1])

    for i in range(n_steps+1):
        
        for j in range(n_samples // batch_size):  # Iterate over batches
            
            with torch.no_grad():
                out_mini_batch1 = model(start[j * batch_size: (j+1) * batch_size].to(device))
                out_mini_batch2 = model(cond_img_batch[j * batch_size: (j+1) * batch_size].to(device))
                scores[i, j * batch_size: (j+1) * batch_size] = measure(out_mini_batch1, out_mini_batch2)
            
        coords = salient_order[:, step * i : step * (i + 1)]

        start = start.reshape(n_samples, 3, HW)
        finish = finish.reshape(n_samples, 3, HW)
        
        for n in range(n_samples):
            start[n, :, coords[n]] = finish[n, :, coords[n]]
        
        start = start.reshape(n_samples, 3, H, W)
        finish = finish.reshape(n_samples, 3, H, W)
        
    scores = scores.mean(1)
    return (scores.sum() - scores[0] / 2 - scores[-1] / 2) / (scores.shape[0] - 1)

def average_drop_increase_simult(model, img1, img2, explanation1, explanation2):
    
    if not torch.is_tensor(explanation1):
        explanation1 = torch.from_numpy(explanation1)
        explanation2 = torch.from_numpy(explanation2)

    size_r, size_c = explanation1.shape[-2], explanation1.shape[-1]
    explanation1, explanation2 = explanation1.reshape(1, 1, size_r, size_c).float(), explanation2.reshape(1, 1, size_r, size_c).float()
        
    measure = nn.CosineSimilarity(dim=-1)   

    saliency_map1 = ((explanation1 - explanation1.min()) / (explanation1.max() - explanation1.min())).to(device)
    masked_image1 = saliency_map1 * img1
    
    saliency_map2 = ((explanation2 - explanation2.min()) / (explanation2.max() - explanation2.min())).to(device)
    masked_image2 = saliency_map2 * img2
    
    with torch.no_grad():
        base_score = measure(model(img1), model(img2))
        score = measure(model(masked_image1), model(masked_image2))
        
    drop = torch.maximum(torch.zeros(1).to(device), base_score - score) / base_score
    increase = (score > base_score).sum()
    return drop, increase

def average_drop_increase_cond(model, img, cond_img, explanation):

    if not torch.is_tensor(explanation):
        explanation = torch.from_numpy(explanation)

    size_r, size_c = explanation.shape[-2], explanation.shape[-1]
    explanation = explanation.reshape(1, 1, size_r, size_c).float()
        
    measure = nn.CosineSimilarity(dim=-1)   

    saliency_map = ((explanation - explanation.min()) / (explanation.max() - explanation.min())).to(device)
    masked_image = saliency_map * img

    with torch.no_grad():
        base_score = measure(model(img), model(cond_img))
        score = measure(model(masked_image), model(cond_img))
        
    drop = torch.maximum(torch.zeros(1).to(device), base_score - score) / base_score
    increase = (score > base_score).sum()
    return drop, increase

def sample_eps_Inf(image, epsilon, N):
    images = np.tile(image, (N, 1, 1, 1))
    dim = images.shape
    return np.random.uniform(-1 * epsilon, epsilon, size=dim)

def max_sensitivity(model, X1, X2, mixed_images, expl1, expl2, exp_fn, use_guided, sen_N = 10, sen_r = 0.2):
    
    """ used for sailency methods (grad x input, Smooth-Grad and Integrated Transforms) """
    
    max_diff1 = -math.inf
    max_diff2 = -math.inf
    joint_max_diff = -math.inf
    
    for _ in range(sen_N):

        if mixed_images is not None:  # integrated
            sample = torch.FloatTensor(sample_eps_Inf(mixed_images[0].cpu().numpy(), sen_r, 1)).to(device)
            mixed_images_noisy = [X + sample for X in mixed_images]

            expl1_noisy, expl2_noisy = exp_fn(guided = use_guided, ssl_model = model, 
                                              mixed_images = mixed_images_noisy, 
                                              blur_output = False)
            
        else:
            sample = torch.FloatTensor(sample_eps_Inf(X1.cpu().numpy(), sen_r, 1)).to(device)
            X1_noisy = X1 + sample
            X2_noisy = X2 + sample
            
            expl1_noisy, expl2_noisy = exp_fn(guided = use_guided, ssl_model = model, 
                                              img1 = X1_noisy, img2 = X2_noisy, 
                                              blur_output = False)
            


        result1 = (expl1 - expl1_noisy).norm() / expl1.norm()
        result2 = (expl2 - expl2_noisy).norm() / expl2.norm()
        
        max_diff1 = max(max_diff1, result1.cpu().numpy())
        max_diff2 = max(max_diff2, result2.cpu().numpy())
        joint_max_diff = (max_diff1 + max_diff2) / 2
        
    return joint_max_diff

def max_sensitivity_cam(model, X1, X2, expl1, expl2, exp_fn, reduction, sen_N = 10, sen_r = 0.2):
    
    """ used for cam methods (Grad-CAM and Interaction-CAM) """

    max_diff1 = -math.inf
    max_diff2 = -math.inf
    joint_max_diff = -math.inf
    
    for _ in range(sen_N):

        sample = torch.FloatTensor(sample_eps_Inf(X1.cpu().numpy(), sen_r, 1)).to(device)
        X1_noisy = X1 + sample
        X2_noisy = X2 + sample
        
        if reduction is None:   # gradcam
            expl1_noisy, expl2_noisy = exp_fn(model, X1_noisy, X2_noisy)
        else:                   # interaction-cam
            expl1_noisy, expl2_noisy = exp_fn(model, X1_noisy, X2_noisy, reduction = reduction)

        result1 = np.linalg.norm(expl1 - expl1_noisy) / np.linalg.norm(expl1)
        result2 = np.linalg.norm(expl2 - expl2_noisy) / np.linalg.norm(expl2)
        
        max_diff1 = max(max_diff1, result1)
        max_diff2 = max(max_diff2, result2)
        joint_max_diff = (max_diff1 + max_diff2) / 2
        
    return joint_max_diff