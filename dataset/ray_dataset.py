from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import gin
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from tqdm import tqdm

from dataset.parsers import get_parser
from dataset.utils import io as data_io
from utils.ray import RayBundle
from utils.render_buffer import RenderBuffer
from utils.tensor_dataclass import TensorDataclass


@gin.configurable()
class RayDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        scene: str = 'lego',
        scene_type: str = 'nerf_synthetic_multiscale',
        split: str = 'train',
        to_world: bool = True,
        num_rays: int = 8192,
        render_bkgd: str = 'white',
        **kwargs
    ):
        super().__init__()
        parser = get_parser(scene_type)
        self.scene_type = scene_type
        data_source = parser(
            base_path=Path(base_path), scene=scene, split=split, **kwargs
        )
        self.training = split.find('train') >= 0

        self.cameras = data_source['cameras']
        logger.info('==> Find {} cameras'.format(len(self.cameras)))
        # parallel loading frames
        if self.scene_type == 'judy_blender':
            self.poses = torch.tensor(np.asarray(data_source['poses'])).float()
            self.width, self.height = self.cameras[0].width, self.cameras[0].height
            self.file_names  = [f['image_filename'] for f in data_source['frames']]
            self.image_names = [f['image_name'] for f in data_source['frames']]
            self.frame_number = len(data_source['frames'])
            self.loss_multi = torch.tensor(1.0)
            #self.ray_bundles = [c.build('cpu') for c in self.cameras] 
        else:
            self.poses = {
                k: torch.tensor(np.asarray(v)).float()  # Nx4x4
                for k, v in data_source["poses"].items()
            }
            self.frames = {}
            for k, cam_frames in data_source['frames'].items():
                with ThreadPoolExecutor(
                    max_workers=min(multiprocessing.cpu_count(), 32)
                ) as executor:
                    frames = list(
                        tqdm(
                            executor.map(
                                lambda f: torch.tensor(
                                    data_io.imread(f['image_filename'])
                                ),
                                cam_frames,
                            ),
                            total=len(cam_frames),
                            dynamic_ncols=True,
                        )
                    )
                self.frames[k] = torch.stack(frames, dim=0)
            self.frame_number = {k: x.shape[0] for k, x in self.frames.items()}
            self.loss_multi = {
                k: torch.tensor([x['lossmult'] for x in v])
                for k, v in data_source['frames'].items()
            }
            self.file_names = {
                k: [x['image_filename'].stem for x in v]
                for k, v in data_source['frames'].items()
            }
            self.ray_bundles = [c.build('cpu') for c in self.cameras]
        self.aabb = torch.tensor(np.asarray(data_source['aabb'])).float()
        self.to_world = to_world
        self.num_rays = num_rays
        self.render_bkgd = render_bkgd
        # try to read a data to initialize RenderBuffer subclass
        self[0]

    def __len__(self):
        if self.training:
            return 10**9  # hack of streaming dataset
        else:
            if self.scene_type == 'judy_blender':
                return self.frame_number
            else:
                return sum([x.shape[0] for k, x in self.poses.items()])

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays


    @torch.no_grad()
    def __getitem__(self, index):
        if self.scene_type == 'judy_blender':
            if self.training: # unevenly sample cameras with different focal length in each batch
                #idx = torch.randint(0, self.frame_number, size=(self.num_rays,))
                #Now we only support single image per batch (do as MultiNeRF-blender does)
                idx = torch.randint(0, len(self.cameras), size=(1,)).item()
                frame = torch.tensor(
                                    data_io.imread(self.file_names[idx])
                                )
                sample_x = torch.randint(
                    0, self.width, size=(self.num_rays,)
                )
                sample_y = torch.randint(
                    0, self.height, size=(self.num_rays,)
                )
                rgb = frame[sample_y, sample_x] #num_rays, 3
                c2w = self.poses[idx][None,...].expand(self.num_rays,-1,-1) #num_rays,4,4
                ray_bundle = self.cameras[idx].build('cpu')
                cam_rays = ray_bundle[sample_y, sample_x] #num_rays
                loss_multi = torch.broadcast_to(self.loss_multi, [self.num_rays, 1]) #scalar
                if 'white' == self.render_bkgd:
                    render_bkgd = torch.ones_like(rgb[..., [-1]])
                elif 'black' == self.render_bkgd:
                    render_bkgd = torch.zeros_like(rgb[..., [-1]])
                elif 'rand' == self.render_bkgd:
                    render_bkgd = torch.rand_like(rgb[..., :3])
                elif 'randn' == self.render_bkgd:
                    render_bkgd = (torch.randn_like(rgb[..., :3]) + 0.5).clamp(
                        0.0, 1.0
                    )
                else:
                    raise NotImplementedError
            else: #now index if the idx of the image in the whole dataset
                ray_bundle = self.cameras[index].build('cpu')
                num_rays = len(ray_bundle)
                idx = torch.ones(size=(num_rays,), dtype=torch.int64) * index
                sample_x, sample_y = torch.meshgrid(
                    torch.arange(self.cameras[index].width),
                    torch.arange(self.cameras[index].height),
                    indexing="xy",
                )
                sample_x = sample_x.reshape(-1)
                sample_y = sample_y.reshape(-1)

                frame = torch.tensor(
                                    data_io.imread(self.file_names[index])
                                )                
                rgb = frame[sample_y, sample_x]
                c2w = self.poses[idx]
                cam_rays = ray_bundle[sample_y, sample_x]
                loss_multi = torch.broadcast_to(self.loss_multi, [num_rays, 1]) 
                render_bkgd = torch.zeros_like(rgb[..., [-1]])  #For my blender

            if self.to_world:
                cam_rays.directions = (
                    c2w[:, :3, :3] @ cam_rays.directions[..., None]
                ).squeeze(-1)
                cam_rays.origins = c2w[:, :3, -1]

            target = RenderBuffer(
                rgb=rgb[..., :3] * rgb[..., [-1]]
                + (1.0 - rgb[..., [-1]]) * render_bkgd,
                render_bkgd=render_bkgd,
                # alpha=rgb[..., [-1]],
                loss_multi=loss_multi,
            )
            if not self.training:
                cam_rays = cam_rays.reshape(
                    (self.height, self.width)
                )
                target = target.reshape(
                    (self.height, self.width)
                )
            outputs = {
                # 'c2w': c2w,
                'cam_rays': cam_rays,
               'target': target,
                # 'idx': idx,
            }
            if not self.training:
                outputs['name'] = self.image_names[index]

        else:
            if self.training:
                rgb, c2w, cam_rays, loss_multi = [], [], [], []
                for cam_idx in range(len(self.cameras)): # [Judy] evenly sample different cameras in each batch
                    num_rays = int(
                        self.num_rays
                        * (1.0 / self.loss_multi[cam_idx][0])
                        / sum([1.0 / v[0] for _, v in self.loss_multi.items()])
                    )
                    idx = torch.randint(
                        0,
                        self.frames[cam_idx].shape[0],
                        size=(num_rays,), #choose num_rays images 
                    )
                    sample_x = torch.randint(
                        0,
                        self.cameras[cam_idx].width,
                        size=(num_rays,),
                    )  # uniform sampling
                    sample_y = torch.randint(
                        0,
                        self.cameras[cam_idx].height,
                        size=(num_rays,),
                    )  # uniform sampling
                    rgb.append(self.frames[cam_idx][idx, sample_y, sample_x])
                    c2w.append(self.poses[cam_idx][idx])
                    cam_rays.append(self.ray_bundles[cam_idx][sample_y, sample_x])
                    loss_multi.append(self.loss_multi[cam_idx][idx, None])
                rgb = torch.cat(rgb, dim=0)
                c2w = torch.cat(c2w, dim=0)
                cam_rays = RayBundle.direct_cat(cam_rays, dim=0)
                loss_multi = torch.cat(loss_multi, dim=0)
                if 'white' == self.render_bkgd:
                    render_bkgd = torch.ones_like(rgb[..., [-1]])
                elif 'black' == self.render_bkgd:
                    render_bkgd = torch.zeros_like(rgb[..., [-1]])
                elif 'rand' == self.render_bkgd:
                    render_bkgd = torch.rand_like(rgb[..., :3])
                elif 'randn' == self.render_bkgd:
                    render_bkgd = (torch.randn_like(rgb[..., :3]) + 0.5).clamp(
                        0.0, 1.0
                    )
                else:
                    raise NotImplementedError

            else:
                for cam_idx, num in self.frame_number.items():
                    if index < num:
                        break
                    index = index - num
                num_rays = len(self.ray_bundles[cam_idx])
                idx = torch.ones(size=(num_rays,), dtype=torch.int64) * index
                sample_x, sample_y = torch.meshgrid(
                    torch.arange(self.cameras[cam_idx].width),
                    torch.arange(self.cameras[cam_idx].height),
                    indexing="xy",
                )
                sample_x = sample_x.reshape(-1)
                sample_y = sample_y.reshape(-1)

                rgb = self.frames[cam_idx][idx, sample_y, sample_x]
                c2w = self.poses[cam_idx][idx]
                cam_rays = self.ray_bundles[cam_idx][sample_y, sample_x]
                loss_multi = self.loss_multi[cam_idx][idx, None]
                render_bkgd = torch.ones_like(rgb[..., [-1]])

            if self.to_world:
                cam_rays.directions = (
                    c2w[:, :3, :3] @ cam_rays.directions[..., None]
                ).squeeze(-1)
                cam_rays.origins = c2w[:, :3, -1]
            target = RenderBuffer(
                rgb=rgb[..., :3] * rgb[..., [-1]]
                + (1.0 - rgb[..., [-1]]) * render_bkgd,
                render_bkgd=render_bkgd,
                # alpha=rgb[..., [-1]],
                loss_multi=loss_multi,
            )
            if not self.training:
                cam_rays = cam_rays.reshape(
                    (self.cameras[cam_idx].height, self.cameras[cam_idx].width)
                )
                target = target.reshape(
                    (self.cameras[cam_idx].height, self.cameras[cam_idx].width)
                )
            outputs = {
                # 'c2w': c2w,
                'cam_rays': cam_rays,
                'target': target,
                # 'idx': idx,
            }
            if not self.training:
                outputs['name'] = self.file_names[cam_idx][index]
        return outputs


def ray_collate(batch):
    res = {k: [] for k in batch[0].keys()}
    for data in batch:
        for k, v in data.items():
            res[k].append(v)
    for k, v in res.items():
        if isinstance(v[0], RenderBuffer) or isinstance(v[0], RayBundle):
            res[k] = TensorDataclass.direct_cat(v, dim=0)
        else:
            res[k] = torch.cat(v, dim=0)
    return res



if __name__ == '__main__':
    training_dataset = RayDataset(
        # '/mnt/bn/wbhu-nerf/Dataset/nerf_synthetic',
        #'/mnt/bn/wbhu-nerf/Dataset/nerf_synthetic_multiscale',
        '/home/yutongchen/3D_projects/data/BlenderNeRF/'
        '1st_scene_v2',
        # 'nerf_synthetic',
        '1st_scene_v2',
        num_rays=1024,
    )
    train_loader = iter(
        DataLoader(
            training_dataset,
            batch_size=1, #8,
            shuffle=False,
            num_workers=0,
            collate_fn=ray_collate,
            pin_memory=True,
            worker_init_fn=None,
            pin_memory_device='cuda',
        )
    )
    # test_dataset = RayDataset(
    #     # '/mnt/bn/wbhu-nerf/Dataset/nerf_synthetic',
    #     '/mnt/bn/wbhu-nerf/Dataset/nerf_synthetic_multiscale',
    #     'lego',
    #     # 'nerf_synthetic',
    #     'nerf_synthetic_multiscale',
    #     num_rays=81920,
    #     split='test',
    # )
    for i in tqdm(range(25000)):
        data = next(train_loader)
        pass
    # for i in tqdm(range(len(test_dataset))):
    #     data = test_dataset[i]
    #     pass
    # pass
