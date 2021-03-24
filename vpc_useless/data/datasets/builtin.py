from .taskonomy.register_with_detectron2 import register_datasets as register_taskonomy_datasets
from .taskonomy.register_with_detectron2 import DATASETS as TASKONOMY_DATASETS
#
# You could import more here
#


DEFAULT_DATASETS_ROOT = "."
register_taskonomy_datasets(TASKONOMY_DATASETS, DEFAULT_DATASETS_ROOT)
# 
# You could register them here
# 