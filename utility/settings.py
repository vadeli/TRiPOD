# settings.py
import torch

def init(str):
    global log_file
    log_file = str
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')