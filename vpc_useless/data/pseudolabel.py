
from detectron2.data.build import build_batch_data_loader
from detectron2.data.samplers import TrainingSampler, InferenceSampler
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MapDataset    
)

from detectron2.utils.registry import Registry


import torch 
from functools import partial
import pdb

def pseudolabel_mapper(x, use_bgr=False):
    '''
        For now, each item in the list is a dict that contains:
            * "image": Tensor, image in (C, H, W) format.
            * "instances": Instances
            * "sem_seg": semantic segmentation ground truth.
            * Other information that's included in the original dicts, such as:
              "height", "width" (int): the output resolution of the model, used in inference.
              See :meth:`postprocess` for details.
    '''
#     new_x = [{} for dict in range(x['num_positive'])]
    x['image'] = x['rgb'] * 255
    if use_bgr:
        channel_dim = 0
        assert x['rgb'].shape[channel_dim] == 3, f"N channels should be 3, but found {x['rgb'].shape}"
        x['image'] = torch.flip(x['image'], dims=(channel_dim,))
    return x
#     for i in range(x['num_positive']):
#         new_x[i]['image'] = x['positive']['rgb'][i] * 255
#         if 'point_info' in x['positive']:
#             new_x[i]['point_info']  = x['positive']['point_info'][i]
#         new_x[i]['mask_valid']  = x['positive']['mask_valid'][i]
#         new_x['height'] = image.shape[1]
#         new_x['width'] = image.shape[2]

    return new_x

def get_unlabeled_dataset_dicts(
    dataset_names,
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
        dataset = constructor(opts)
        datasets.append(dataset)

    for dataset_name, dataset in zip(dataset_names, datasets):
        assert len(datasets), "Dataset '{}' is empty!".format(dataset_name)

    return torch.utils.data.ConcatDataset(datasets)


def concatenating_batcher(*args, **kwargs):
    batcher = build_batch_data_loader(*args, **kwargs)
    concatenate = partial(sum, start=[])
    return map(concatenate, batcher)




def build_unlabeled_for_pseudolabeling_loader(cfg, mapper=None):
    dataset = get_unlabeled_dataset_dicts(cfg.DATASETS.CONSISTENCY_TRAIN)

    if mapper is None:
        mapper = partial(pseudolabel_mapper, use_bgr=(cfg.INPUT.FORMAT.upper()=='BGR'))
    dataset = MapDataset(dataset, mapper)


    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    num_workers = 8
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler,
                                                          cfg.SOLVER.IMS_PER_BATCH,
                                                          drop_last=False)
    
    data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
        )

    return data_loader

def build_pseudolabel_consistency_loader(cfg, mapper=None):
    dataset = get_consistency_dataset_dicts(cfg.DATASETS.CONSISTENCY_TRAIN)

    if mapper is None:
        mapper = partial(pseudolabel_mapper, use_bgr=(cfg.INPUT.FORMAT.upper()=='BGR'))
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


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch

#     build_batch_data_loader(
#         dataset,
#         sampler,
#         cfg.SOLVER.CONSISTENCY_POINTS_PER_BATCH, # = num gpus
#         aspect_ratio_grouping=False,
#         num_workers=num_workers,
#     )



