from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import torch


def extract_feature(img):
    weights = ResNet50_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    model = resnet50(weights=weights)
    print("loaded")

    model.eval()

    img_transformed = preprocess(img)
    batch_t = torch.unsqueeze(img_transformed, 0)
    print(batch_t.shape)
    output = model(batch_t)
    print(output.shape)