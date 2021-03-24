import seaborn as sns
import torch
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

# Visualise SS
def rgb2gray(rgb):
    return torch.tensor(np.dot(rgb[:,:,:3], [0.299, 0.587, 0.114])) #/ 255.0

DEFAULT_LABELS = ['uncertain',
 'background',
 'bottle',
 'chair',
 'couch',
 'potted_plant',
 'bed',
 'dining_table',
 'toilet',
 'tv',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase']


def colorize_semseg(pred, cmap=None, uncertain_class=1):
    if cmap is None:
        cmap = cm.get_cmap('tab20')
    cmap_length = 20
    p = torch.tensor(pred).max(dim=-1)[1].cpu().numpy()
    ss = cmap(p % cmap_length)[...,:3]
#     ss[p == uncertain_class] = 0.0
    return ss

def show_semseg_legend(target_predictions, labels=DEFAULT_LABELS, cmap='tab20', show=True):
    vals, counts = torch.stack(target_predictions).squeeze().cpu().max(dim=-1)[1].unique(return_counts=True)
    pal = np.array(sns.color_palette(cmap))
    brightness_factor = 8
    pal[vals] *= brightness_factor
    pal /= brightness_factor
    sns.palplot(pal[:len(labels)])
    plt.xticks(range(len(labels)), labels)
    if show:
        plt.show()
    
def alpha_blend(alpha):
    assert 0.0 <= alpha and alpha <= 1.0, 'alpha must be between 0 and 1'

    def _alpha_blend(x, y, mask=None):
        if mask is None:
            mask = 1.0
        return (x * alpha + y * (1 - alpha)) * mask
    return _alpha_blend

def plot_consistent_preds(rgb, original_preds, consistent_preds, masks, names=None,
                          scale=4, show=True, batch_idx=0, combiner=None ):
    
    ''' 
        rgb: a batch of B x W x H x C images
        original_preds: a batch of B x W x H images
        consistent_preds: a batch of B x W x H images
        names: names for each column
        combiner(RGB, Predictions) -> combined image
        
    '''
    if combiner is None:
        combiner = alpha_blend(0.4)

    n_views = len(rgb)
    assert n_views == len(original_preds) and n_views == len(consistent_preds)  
    

    fig, ax = plt.subplots(nrows=2, ncols=n_views, figsize = (scale * n_views, 2 * scale))

    for j, view in enumerate(names):
        for i, row in enumerate(ax):
            row_name = ['Original', 'Consistent'][i]
            p = [original_preds, consistent_preds][i][j]
#             print(f'plot_consistent_preds right before_combiner {row_name} :{p.shape}')
            mask = masks[j] if masks is not None else None
            im = combiner(rgb[j], p, mask)
#             p * alpha + np.array(rgb[j]) * (1 - alpha)
            row[j].imshow(im)
            if j == 0:
                row[j].set_ylabel(['Original', 'Consistent'][i])
            if i == 0:
                row[j].set_title(names[j])

    plt.tight_layout()
    if show:
        plt.show()

        
        
        

import einops as ein
import torch.nn.functional as F
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.visualizer import Visualizer



def visualize_consistency(viewpoint_batch, stacked_predictions, target_predictions, cfg, size=None, masks=None, show=False):
    if size is None:
        size = viewpoint_batch[0]['image'].shape[1:]
    rgbs = torch.stack([v['image'].detach().cpu() for v in viewpoint_batch])
    
    stacked_predictions = ein.rearrange(stacked_predictions, 'b c w h -> b w h c')
    orig_preds = stacked_predictions.cpu().detach()

    target_predictions = F.interpolate(
            target_predictions,
            size,
            mode='nearest').long()
    target_preds = orig_preds # target_predictions.cpu().detach()
    target_preds = ein.rearrange(target_predictions.cpu().detach(), 'b c w h -> b w h c')

    
    def vizier(rgb, semseg, mask=None, alpha=0.6):
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        chan_last = lambda x: x.unsqueeze(-1).transpose(-1, 0).squeeze(0)

        # Pytorch tensor is in (C, H, W) format
        # img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
        # img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)
        rgb = torch.tensor(rgb)
        img = chan_last(torch.tensor(rgb)).cpu().detach().numpy()
        img = convert_image_to_rgb(img, cfg.INPUT.FORMAT)

        res = torch.tensor(semseg).argmax(dim=-1).double().long()

        scale = 1.0
        visualizer = Visualizer(img, metadata=metadata, scale=scale)
        vis = visualizer.draw_sem_seg(res, area_threshold=None, alpha=alpha)
        im = vis.get_image()
        if mask is not None:
            im = np.uint8(mask * im)
        return im

    plot_consistent_preds(rgbs, orig_preds, target_preds, masks, scale=10.0,
                         names=[v['point_info']['view_id'] for v in viewpoint_batch],
                         show=show, combiner=vizier)

