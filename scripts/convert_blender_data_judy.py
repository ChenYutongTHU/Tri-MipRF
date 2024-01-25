import json
import os
from os import path

from absl import app
from absl import flags
import numpy as np
from PIL import Image

FLAGS = flags.FLAGS

flags.DEFINE_string('blenderdir', None, 'Base directory for all Blender data.')



def convert_to_nerfdata(basedir, splits):
    bigmeta = {}
    # Foreach split in the dataset
    for split in splits:
        print('Split', split, end=' ')

        f = '{}.json'.format(split)
        with open(path.join(basedir, f), 'r') as fp:
            meta_input = json.load(fp)
        N = len(meta_input['frames'])
        print('#{} images'.format(N))
        keys = meta_input['frames'][0].keys()
        key2list = {k:[f[k] for f in meta_input['frames']] for k in keys}
        print('We assume all images have the same size.')
        example_file = os.path.join(basedir, meta_input['frames'][0]['file_path'])
        image = np.array(Image.open(example_file), dtype=np.float32) / 255.0
        h, w = image.shape[0], image.shape[1]

        meta = {}
        meta['cam2world'] = key2list['transform_matrix']
        meta['focal'] = []
        for camera_angle_x in key2list['camera_angle_x']:
            camera_angle_x = float(camera_angle_x)
            meta['focal'].append(0.5 * w / np.tan(0.5 * camera_angle_x))
        meta['image_names'] = [n.replace('.01png','') for n in key2list['image_name']]
        meta['file_path'] = key2list['file_path']
        meta['width'] = [w for _ in range(N)]
        meta['height'] = [h for _ in range(N)]
        meta['near'] = key2list['near']
        meta['far'] = key2list['far']
        meta['lossmult'] = [1.0 for _ in range(N)]
        meta['label'] = [0 for _ in range(N)]
        fx = np.array(meta['focal']) #N,
        fy = np.array(meta['focal'])
        cx = np.array(meta['width']) * 0.5
        cy = np.array(meta['height']) * 0.5
        arr0 = np.zeros_like(cx)
        arr1 = np.ones_like(cx)
        k_inv = np.array(
            [
                [arr1 / fx, arr0, -cx / fx],
                [arr0, -arr1 / fy, cy / fy],
                [arr0, arr0, -arr1],
            ]
        )
        k_inv = np.moveaxis(k_inv, -1, 0)
        meta['pix2cam'] = k_inv.tolist()

        bigmeta[split] = meta

    for k in bigmeta:
        for j in bigmeta[k]:
            print(k, j, type(bigmeta[k][j]), np.array(bigmeta[k][j]).shape)

    jsonfile = os.path.join(basedir, 'metadata.json')
    with open(jsonfile, 'w') as f:
        json.dump(bigmeta, f, ensure_ascii=False, indent=4)


def main(unused_argv):

    blenderdir = FLAGS.blenderdir

    # Unlike the original script, 
    # we do not downsample images, 
    # but only convert the format of metadata and place it in the original dir
    convert_to_nerfdata(blenderdir, splits=[
        'train_single-scale',
        'train_multi-scale_feet',
        'train_multi-scale_lens',
        'test_single-scale',
        'test_zoom-in_feet',
        'test_zoom-in_lens',])


if __name__ == '__main__':
    app.run(main)
