from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import os
import torch
import numpy as np

def extract_feature(dir):
    weights = ResNet50_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    model = resnet50(weights=weights)
    print("ResNet loaded!")

    feature_list = []

    model.eval()

    count = 0
    # for subdir in os.listdir(dir):
    #     print(subdir)
    #     for image in os.listdir(dir+"/"+subdir):
    #         img = Image.open(dir+"/"+subdir+"/"+image)
    #         img_transformed = preprocess(img)
    #         batch_t = torch.unsqueeze(img_transformed, 0)
    #         output = model(batch_t)
    #         output = np.squeeze(output.detach().numpy())
    #         feature_list.append(output)
    #     print(count)

    for image in os.listdir(dir):
        img = Image.open(dir+image)
        img = img.convert('RGB')
        print(img)
        img_transformed = preprocess(img)
        batch_t = torch.unsqueeze(img_transformed, 0)
        output = model(batch_t)
        output = np.squeeze(output.detach().numpy())
        feature_list.append(output)
        print(count)
        count = count + 1
        

    return feature_list