import cv2
from tqdm import tqdm
import os 

folder = "/data9/leolxu/uniform_illumination_unet/渲染数据集/8"
for name in tqdm(os.listdir(folder)):

    img = cv2.imread(os.path.join(folder, name), cv2.IMREAD_UNCHANGED)

    # print(img.shape)
    H,W,C = img.shape

    ratio = 0.1

    _img = img[:int(H-H*ratio),int(W*ratio/2):int(W-W*ratio/2),:]
    cv2.imwrite(os.path.join(folder, name), _img)