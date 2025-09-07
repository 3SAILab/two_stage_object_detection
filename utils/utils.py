import numpy as np
import logging
import random
import torch
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def update_ema(current_value, ema_alpha, last_ema=None):
    if last_ema is None:
        return current_value
    return ema_alpha * current_value + (1 - ema_alpha) * last_ema

def show_model_flops_and_params(model):
    from thop import profile
    import json

    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/config.json")) as f:
        config = json.load(f)
    device = config['device']

    model.eval()
    input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(input, ), verbose=False)
    logging.info(f"➡️  FLOPs = {str(flops/1000**3)} G")
    logging.info(f"➡️  Params = {str(params/1000**2)} M")
    return flops, params