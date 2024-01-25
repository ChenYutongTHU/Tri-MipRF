from pathlib import Path
import numpy as np
from tqdm import tqdm

import dataset.utils.io as data_io
from dataset.utils.cameras import PinholeCamera


def load_data(base_path: Path, scene: str, split: str):
    # ipdb.set_trace()
    data_path = base_path / scene
    meta_path = data_path / 'metadata.json'

    splits = ['train', 'val'] if split == "trainval" else [split]
    meta = None
    for s in splits:
        m = data_io.load_from_json(meta_path)[s]
        if meta is None:
            meta = m
        else:
            for k, v in meta.items():
                v.extend(m[k])

    pix2cam = meta['pix2cam']
    poses = meta['cam2world']
    image_width = meta['width']
    image_height = meta['height']
    lossmult = meta['lossmult']
    cameras = []
    for i in range(len(pix2cam)):
        k = np.linalg.inv(pix2cam[i])
        fx = k[0, 0]
        fy = -k[1, 1]
        cx = -k[0, 2]
        cy = -k[1, 2]
        cam = PinholeCamera(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=image_width[i],
            height=image_height[i],
            # loss_multi=lossmult[i],
        )
        cameras.append(cam)

    frames = []
    for image_name, frame in tqdm(zip(meta['image_names'], meta['file_path'])):
        fname = data_path / frame
        frames.append(
            {   
                'image_name': image_name, 
                'image_filename': fname,
                'lossmult': 1.0,
            }
        )

    #aabb = np.array([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5]) #??
    aabb = np.array([-5.0, -5.0, -5.0, 5.0, 5.0, 5.0])  # ??
    outputs = {
        'frames': frames, # len(frames) // 4
        'poses': poses,
        'cameras': cameras,
        'aabb': aabb,
    }
    return outputs

