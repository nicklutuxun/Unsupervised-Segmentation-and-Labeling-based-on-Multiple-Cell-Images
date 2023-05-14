from torchvision.models import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights, vgg19_bn, VGG19_BN_Weights
from PIL import Image
import os
import torch
import numpy as np
from torchvision import transforms

def extract_feature(dir):
    weights = ResNet152_Weights.IMAGENET1K_V2
    # preprocess = weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = resnet152(weights=weights)
    print("ResNet loaded!")

    feature_list = []

    model.eval()

    count = 0

    for image in os.listdir(dir):
        # image = "57.png"
        img = Image.open(dir+image)
        img = img.convert('RGB')
        img = np.asarray(img)
        # plt.subplot(1,3,1)
        # plt.imshow(img)
        img_transformed = pad_img(img)
        img = Image.fromarray(img_transformed)
        # plt.subplot(1,3,2)
        # plt.imshow(img)
        img = preprocess(img)
        # img = np.transpose(img.numpy(), (1,2,0))
        # plt.subplot(1,3,3)
        # plt.imshow(img)

        # break
        batch_t = torch.unsqueeze(img, 0)
        output = model(batch_t)
        output = np.squeeze(output.detach().numpy())
        feature_list.append(output)
        print("Finished: ", count)
        count = count + 1
        
    return feature_list


def extract_feature_img(img, classifier, preprocess):
    img = img.convert('RGB')
    img = np.asarray(img)
    img_transformed = pad_img(img)
    img_transformed = Image.fromarray(img_transformed)
    img_transformed = preprocess(img_transformed)
    batch_t = torch.unsqueeze(img_transformed, 0)
    output = classifier(batch_t)
    output = np.squeeze(output.detach().numpy())

    return output


def pad_img(img):
    h, w, c = img.shape
    max_dim = max(h,w)
    # max_dim = 416
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

    return padded_img