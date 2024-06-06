import os 
import cv2
from collections import defaultdict 
import shutil
from tqdm import tqdm
source_folder = "/data9/leolxu/uniform_illumination_unet/渲染数据集"
target_folder = "/data9/leolxu/uniform_illumination_unet/result_blender"


global_index = -1
name = {}
for subdir in ["1","2","3","4","5","6", "7", "8",]:
    name[subdir] = defaultdict(list) 
    for img in os.listdir(os.path.join(source_folder, subdir)):
        name[subdir][img.split("_")[0]].append(img)

    for k,v in tqdm(name[subdir].items()):
        global_index += 1
        new_prefix = "%03d"%global_index
        target_sub_folder = os.path.join(target_folder, new_prefix)
        os.makedirs(target_sub_folder, exist_ok=True)
        for raw_name in v:
            raw_path = os.path.join(source_folder, subdir, raw_name)
            if "_no_back.png" in raw_name:
                img = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
                save_path = os.path.join(target_sub_folder, "matting.png")
                cv2.imwrite(save_path, img[:,:,3:4])

            elif "_back.jpg" in raw_path:
                save_path = os.path.join(target_sub_folder, "no_light.jpg")
                shutil.copyfile(raw_path, save_path)
            else:
                new_name = raw_name.replace(k + "_", new_prefix + "_")
                save_path = os.path.join(target_sub_folder, new_name)
                shutil.copyfile(raw_path, save_path)          



    print(subdir, len(name[subdir]), name[subdir]['0'])



