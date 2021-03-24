
from detectron2.data.build import build_batch_data_loader
from detectron2.data.samplers import TrainingSampler
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MapDataset    
)

from detectron2.utils.registry import Registry

CONSISTENCY_MESH_REGISTRY = Registry("CONSISTENCY_MESH")
CONSISTENCY_MESH_REGISTRY.__doc__ = """
Registry for meshes used for consistency.
The registered object should be a Pytorch3D mesh.
"""


import torch 
from functools import partial
import pdb

def consistency_mapper(x, use_bgr=False):
    '''
        For now, each item in the list is a dict that contains:
            * "image": Tensor, image in (C, H, W) format.
            * "instances": Instances
            * "sem_seg": semantic segmentation ground truth.
            * Other information that's included in the original dicts, such as:
              "height", "width" (int): the output resolution of the model, used in inference.
              See :meth:`postprocess` for details.
    '''
    new_x = [{} for dict in range(x['num_positive'])]
    images = x['positive']['rgb']
    if use_bgr:
        channel_dim = 1
        assert x['positive']['rgb'].shape[channel_dim] == 3, f"N channels should be 3, but found {x['positive']['rgb'].shape}"
        images = torch.flip(images, dims=(channel_dim,))
    for i in range(x['num_positive']):
        new_x[i]['image'] = images[i] * 255
        new_x[i]['point_info']  = x['positive']['point_info'][i]
#         new_x[i]['mask_valid']  = x['positive']['mask_valid'][i]
#         new_x['height'] = image.shape[1]
#         new_x['width'] = image.shape[2]

    return new_x

def get_consistency_dataset_dicts(
    dataset_names,
    max_views=None,
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.
    Args:
        dataset_names (list[str]): a list of dataset names
            that match each dataset in `dataset_names`.
    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    assert len(dataset_names)

    dataset_opts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    
    datasets = []
    for constructor, opts in dataset_opts:
        if max_views is not None:
            opts.max_views = max_views
        dataset = constructor(opts)
        for mesh_key, mesh in dataset.meshes.items():
            mesh.__name__ = mesh_key
            CONSISTENCY_MESH_REGISTRY.register(mesh)
        datasets.append(dataset)

    for dataset_name, dataset in zip(dataset_names, datasets):
        assert len(datasets), "Dataset '{}' is empty!".format(dataset_name)

    return torch.utils.data.ConcatDataset(datasets)


def concatenating_batcher(*args, **kwargs):
    batcher = build_batch_data_loader(*args, **kwargs)
    concatenate = partial(sum, start=[])
    return map(concatenate, batcher)


def build_consistency_train_loader(cfg, mapper=None):
    dataset = get_consistency_dataset_dicts(cfg.DATASETS.CONSISTENCY_TRAIN)

    if mapper is None:
        mapper = partial(consistency_mapper, use_bgr=(cfg.INPUT.FORMAT.upper()=='BGR'))
    dataset = MapDataset(dataset, mapper)

    sampler = TrainingSampler(len(dataset))

    num_workers = float(cfg.SOLVER.CONSISTENCY_POINTS_PER_BATCH) / float(cfg.SOLVER.IMS_PER_BATCH) * cfg.DATALOADER.NUM_WORKERS
    num_workers = int(max(1.0, num_workers))

    return concatenating_batcher(
        dataset,
        sampler,
        cfg.SOLVER.CONSISTENCY_POINTS_PER_BATCH, # = num gpus
        aspect_ratio_grouping=False,
        num_workers=num_workers,
    )






#     build_batch_data_loader(
#         dataset,
#         sampler,
#         cfg.SOLVER.CONSISTENCY_POINTS_PER_BATCH, # = num gpus
#         aspect_ratio_grouping=False,
#         num_workers=num_workers,
#     )


