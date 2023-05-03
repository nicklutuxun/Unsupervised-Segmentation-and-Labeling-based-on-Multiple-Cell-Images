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
    for subdir in os.listdir(dir):
        print(subdir)
        for image in os.listdir(dir+"/"+subdir):
            img = Image.open(dir+"/"+subdir+"/"+image)
            # img = cv2.imread("./images/"+subdir+"/"+image, )
            # print(img.shape)
            img_transformed = preprocess(img)
            batch_t = torch.unsqueeze(img_transformed, 0)
            output = model(batch_t)
            output = np.squeeze(output.detach().numpy())
            # torch.save(output, f'./features/tensor_{image}.pt')
            feature_list.append(output)
        print(count)
        # count = count+1
        # if count == 2:
        #     break
        

    return feature_list