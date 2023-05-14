import numpy as np
from PIL import Image
import os


def crop_cells(dir):
    os.system("rm -rf ./cell\ images && mkdir cell\ images")
    count = 1
    for i in range(1,21):
        img = np.asarray(Image.open(f"./{dir}/{i}_rgb.tif").convert('RGB'))
        seg = np.asarray(Image.open(f"./{dir}/{i}_seg.png").convert('L'))
        classes = np.unique(seg)

        for j in range(1,len(classes)-1):    
            c = (seg == classes[j+1])
            mask = np.dstack((c,c,c))
            cell = img*mask

            mask = (cell).any(2)
            cell = cell[np.ix_(mask.any(1),mask.any(0))]

            cell_img = Image.fromarray(cell, "RGB")
            cell_img.save(f"./cell images/{count}.png")
            count = count + 1
        print("Finishied: ", i)