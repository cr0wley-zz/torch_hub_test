# import the necessary packages
import torch
from pyimagesearch import mlp

# entry point
def custom_model(pretrained=False, *args, **kwargs):
    # initialize model instance
    # load weights from path
    model = mlp.get_training_model()
    model.load_state_dict(torch.load("model_wt.pth"))
    return model



