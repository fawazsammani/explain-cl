import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image 
import random
import cv2
import io
from ssl_models.simclr2 import get_simclr2_model
from ssl_models.barlow_twins import get_barlow_twins_model
from ssl_models.simsiam import get_simsiam
from ssl_models.dino import get_dino_model_without_loss, get_dino_model_with_loss

def get_ssl_model(network, variant):
    
    if network == 'simclrv2':
        if variant == '1x':
            ssl_model = get_simclr2_model('r50_1x_sk0_ema.pth').eval()
        else:
            ssl_model = get_simclr2_model('r50_2x_sk0_ema.pth').eval()
    elif network == 'barlow_twins':
        ssl_model = get_barlow_twins_model().eval()  
    elif network == 'simsiam':
        ssl_model = get_simsiam().eval()
    elif network == 'dino':
        ssl_model = get_dino_model_without_loss().eval()
    elif network == 'dino+loss':
        ssl_model, dino_score = get_dino_model_with_loss()
        ssl_model = ssl_model.eval()
        
    return ssl_model

def overlay_heatmap(img, heatmap, denormalize = False):
    loaded_img = img.squeeze(0).cpu().numpy().transpose((1, 2, 0))
    
    if denormalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        loaded_img = std * loaded_img + mean 
                  
    loaded_img = (loaded_img.clip(0, 1) * 255).astype(np.uint8)
    cam = heatmap / heatmap.max()
    cam = cv2.resize(cam, (224, 224))
    cam = np.uint8(255 * cam)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)   # jet: blue --> red
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    added_image = cv2.addWeighted(cam, 0.5, loaded_img, 0.5, 0)
    return added_image

def viz_map(img_path, heatmap):
    "For pixel invariance"
    img = np.array(Image.open(img_path).resize((224,224))) if isinstance(img_path, str) else np.array(img_path.resize((224,224)))
    width, height, _ = img.shape
    cam = heatmap.detach().cpu().numpy()
    cam = cam / cam.max()
    cam = cv2.resize(cam, (height, width))
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    added_image = cv2.addWeighted(heatmap, 0.5, img, 0.7, 0)
    return added_image

def show_image(x, squeeze = True, denormalize = False):
    
    if squeeze:
        x = x.squeeze(0)
        
    x = x.cpu().numpy().transpose((1, 2, 0))
    
    if denormalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        x = std * x + mean 
    
    return x.clip(0, 1)

def deprocess(inp, to_numpy = True, to_PIL = False, denormalize = False):
    
    if to_numpy:
        inp = inp.detach().cpu().numpy()
    
    inp = inp.squeeze(0).transpose((1, 2, 0))
           
    if denormalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean 
           
    inp = (inp.clip(0, 1) * 255).astype(np.uint8)
           
    if to_PIL:
        return Image.fromarray(inp)
    return inp

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    return img
