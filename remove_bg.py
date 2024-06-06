from pathlib import Path
from rembg import remove, new_session
from tqdm import tqdm

model_name = ["u2net", "u2net_human_seg", "u2net_cloth_seg", "snet-general-use", "isnet-anime"][0]

session = new_session(model_name, providers=['CUDAExecutionProvider'])

for file in tqdm(Path("去光照测试集_JUN").glob('*.jpg')):
    input_path = str(file)
    output_path = str(file.parent / Path("mask") / (file.stem  + "_%s.png"%model_name))
    
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input, session=session, alpha_matting=True, alpha_matting_foreground_threshold=270,alpha_matting_background_threshold=20, alpha_matting_erode_size=11, post_process_mask=True)
            o.write(output)