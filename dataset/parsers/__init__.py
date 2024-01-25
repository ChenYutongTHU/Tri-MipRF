from typing import Callable

from . import nerf_synthetic
from . import nerf_synthetic_multiscale
from . import judy_blender


def get_parser(parser_name: str) -> Callable:
    if 'nerf_synthetic' == parser_name:
        return nerf_synthetic.load_data
    elif 'nerf_synthetic_multiscale' == parser_name:
        return nerf_synthetic_multiscale.load_data
    elif 'judy_blender' == parser_name:
        return judy_blender.load_data
    else:
        raise NotImplementedError
