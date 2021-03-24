from dataclasses import dataclass
import os
from typing import Any, Dict, Iterable, List, Optional

from . import TaskonomyDataset, TaskonomyMultiViewDataset

from detectron2.data import DatasetCatalog, MetadataCatalog


@dataclass
class TaskonomyMultiViewDatasetInfo:
    name: str
    opts: TaskonomyMultiViewDataset.Options


@dataclass
class TaskonomyDatasetInfo:
    name: str
    opts: TaskonomyDataset.Options


DATASETS = [
    # Pseudolabels
    TaskonomyDatasetInfo(
        name='taskonomy-tiny-train',
        opts=TaskonomyDataset.Options(
            data_path='datasets/taskonomy',
            tasks=['rgb', 'point_info', 'mask_valid'],
            buildings='tiny-train',
            image_size=512,
        )
    ),
    TaskonomyDatasetInfo(
        name='taskonomy-tiny-test',
        opts=TaskonomyDataset.Options(
            data_path='datasets/taskonomy',
            tasks=['rgb', 'point_info', 'mask_valid'],
            buildings='tiny-val',
            image_size=512,
        )
    ),
    TaskonomyDatasetInfo(
        name='taskonomy-med-train',
        opts=TaskonomyDataset.Options(
            data_path='datasets/taskonomy',
            tasks=['rgb', 'point_info', 'mask_valid'],
            buildings='medium-train',
            image_size=512,
        )
    ),
    TaskonomyDatasetInfo(
        name='taskonomy-med-test',
        opts=TaskonomyDataset.Options(
            data_path='datasets/taskonomy',
            tasks=['rgb', 'point_info', 'mask_valid'],
            buildings='medium-val',
            image_size=512,
        )
    ),
    TaskonomyDatasetInfo(
        name='taskonomy-debug-train',
        opts=TaskonomyDataset.Options(
            data_path='datasets/taskonomy',
            tasks=['rgb', 'point_info', 'mask_valid'],
            buildings=['onaga', 'benicia'],
            image_size=800,
        )
    ),
    TaskonomyDatasetInfo(
        name='taskonomy-debug-val',
        opts=TaskonomyDataset.Options(
            data_path='datasets/taskonomy',
            tasks=['rgb', 'point_info', 'mask_valid'],
            buildings=['onaga'],
            image_size=512,
        )
    ),
    
    
    # Multiview
    TaskonomyMultiViewDatasetInfo(
        name='taskonomy-multiview-tiny-train',
        opts=TaskonomyMultiViewDataset.Options(
            data_path='datasets/taskonomy',
            tasks=['rgb', 'point_info'],
            buildings='tiny-train',
            num_positive=3,
            num_negative=0,  
            load_building_meshes=True,
        )
    ),
    TaskonomyMultiViewDatasetInfo(
        name='taskonomy-multiview-tiny-test',
        opts=TaskonomyMultiViewDataset.Options(
            data_path='datasets/taskonomy',
            tasks=['rgb', 'point_info'],
            buildings='tiny-val',
            num_positive=3,
            num_negative=0,  
            load_building_meshes=True,
        )
    ),
    TaskonomyMultiViewDatasetInfo(
        name='taskonomy-multiview-med-train',
        opts=TaskonomyMultiViewDataset.Options(
            data_path='datasets/taskonomy',
            tasks=['rgb', 'point_info'],
            buildings='medium-train',
            num_positive=3,
            num_negative=0,  
            load_building_meshes=True,
        )
    ),
    TaskonomyMultiViewDatasetInfo(
        name='taskonomy-multiview-med-test',
        opts=TaskonomyMultiViewDataset.Options(
            data_path='datasets/taskonomy',
            tasks=['rgb', 'point_info'],
            buildings='medium-val',
            num_positive=3,
            num_negative=0,  
            load_building_meshes=True,
        )
    ),
    TaskonomyMultiViewDatasetInfo(
        name='taskonomy-multiview-debug-train',
        opts=TaskonomyMultiViewDataset.Options(
            data_path='datasets/taskonomy',
            tasks=['rgb', 'point_info'],
            buildings=['onaga', 'benicia'],
            num_positive=3,
            num_negative=0,  
            load_building_meshes=True,
        )
    ),
    TaskonomyMultiViewDatasetInfo(
        name='taskonomy-multiview-debug-val',
        opts=TaskonomyMultiViewDataset.Options(
            data_path='datasets/taskonomy',
            tasks=['rgb', 'point_info'],
            buildings=['onaga'],
            num_positive=3,
            num_negative=0,    
            load_building_meshes=True,            
        )
    ),
]


def register_dataset(dataset_info: TaskonomyMultiViewDatasetInfo, datasets_root: Optional[str]):
    """
    Registers provided Taskonomy dataset
    Args:
    dataset_info: TaskonomyDatasetInfo
        Dataset data
    datasets_root: Optional[os.PathLike]
        Datasets root folder (default: None)
    """

    dataset_info.opts.data_path = os.path.join(
        datasets_root,
        dataset_info.opts.data_path
    )
    
    def get_taskonomy_dataset():
        if isinstance(dataset_info.opts, TaskonomyMultiViewDataset.Options):
            return (TaskonomyMultiViewDataset, dataset_info.opts)
        elif isinstance(dataset_info.opts, TaskonomyDataset.Options):
            return (TaskonomyDataset, dataset_info.opts)
        else:
            raise NotImplementedError

    DatasetCatalog.register(dataset_info.name, get_taskonomy_dataset)
    MetadataCatalog.get(dataset_info.name).set(**dataset_info.opts.__dict__)


def register_datasets(
    datasets_data: Iterable[TaskonomyMultiViewDatasetInfo], datasets_root: Optional[os.PathLike] = None
):
    """
    Registers provided COCO DensePose datasets
    Args:
    datasets_data: Iterable[CocoDatasetInfo]
        An iterable of dataset datas
    datasets_root: Optional[os.PathLike]
        Datasets root folder (default: None)
    """
    for dataset_data in datasets_data:
        register_dataset(dataset_data, datasets_root)