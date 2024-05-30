import torch
import torchvision as vision
import numpy

def loadPTmodel():
    # Initialize model with the best available weights
    weights = vision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = vision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights,box_score_thresh=0.95)
    model.eval()

    return model, weights

def detectPT(img,model,weights):
    # Reshape image and convert to a tensor.
    X = numpy.asarray(img)
    X_torch1 = torch.from_numpy(numpy.copy(X))
    if X_torch1.ndim==3:
      X_torch = torch.permute(X_torch1,(2,0,1))
    elif X_torch1.ndim==4:
      X_torch = torch.permute(X_torch1,(3,2,0,1))
    # Initialize the inference transforms
    preprocess = weights.transforms()
    # Apply inference preprocessing transforms
    batch = [preprocess(X_torch)]
    # Use the model 
    if X_torch.ndim==3:
      prediction = model(batch)[0]
    elif X_torch.ndim==4:
      prediction = model(list(batch[0]))

    return prediction
