import numpy as np
from extract_feature import extract_feature_img
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from torchvision.models import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights, vgg19_bn, VGG19_BN_Weights

weights = ResNet152_Weights.IMAGENET1K_V2
preprocess = weights.transforms()
model = resnet152(weights=weights)
print("ResNet loaded!")

model.eval()

def overlay(img, seg, classifier):
    classes = np.unique(seg)
    overlay_masks = []
    centers = []

    for i in range(len(classes)-1):    
        c = (seg == classes[i+1])
        mask = np.dstack((c,c,c))
        cell = img*mask

        mask = (cell).any(2)
        cell = cell[np.ix_(mask.any(1),mask.any(0))]

        cell_img = Image.fromarray(cell, "RGB")
        feature = extract_feature_img(cell_img, model, preprocess).reshape(1, -1).astype(np.float64)
        label = classifier.predict(feature)
        overlay_mask = c * (label+1)
        overlay_masks.append(overlay_mask)
        centers.append((get_center(c),label[0]))
    
    overlay_mask = np.sum(overlay_masks, axis=(0))

    cmap = plt.cm.get_cmap("plasma")
    cmap.set_under(alpha=0)
    plt.figure(figsize = (20,20))
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.subplot(1,3,2)
    plt.imshow(seg, cmap="gray")
    plt.subplot(1,3,3)
    plt.imshow(img, interpolation='none')
    plt.imshow(overlay_mask, interpolation='none', alpha=0.75, cmap=cmap, vmin=0.1)
    for i in centers:
        plt.text(i[0][1], i[0][0], i[1], color="white")


def get_center(mask):
    x_min = mask.any(1).argmax()
    y_min = mask.any(0).argmax()
    x_max = mask.shape[0] - 1 - np.rot90(mask, 2).any(1).argmax()
    y_max = mask.shape[1] - 1 - np.rot90(mask, 2).any(0).argmax()

    x_center = int((x_min + x_max) / 2) + 18
    y_center = int((y_min + y_max) / 2) - 9
    return x_center, y_center

mask = np.array([[0,0,0,1,0,0], [0,1,1,1,1,0], [0,0,1,1,1,0], [0,0,0,1,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0]])
print(get_center(mask))






