# import the necessary packages
import torch
from pyimagesearch import mlp
import os

# initialize the model weights path
MODEL_PATH = os.path.join("output", "model_wt.pth")

# entry point
def custom_model():
    # initialize model instance
    # load weights from path
    model = mlp.get_training_model()
    model.load_state_dict(torch.load(MODEL_PATH))
    return model



