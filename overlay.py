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
    
    overlay_mask = np.sum(overlay_masks, axis=(0))

    cmap = plt.cm.get_cmap("plasma")
    cmap.set_under(alpha=0)
    plt.figure(figsize = (10,10))
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.subplot(1,3,2)
    plt.imshow(seg, cmap="gray")
    plt.subplot(1,3,3)
    plt.imshow(img, interpolation='none')
    plt.imshow(overlay_mask, interpolation='none', alpha=0.75, cmap=cmap, vmin=0.1)






