import einops as ein
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer import (
    BlendParams,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    hard_rgb_blend
    )
from pytorch3d.structures import Meshes
import torch
from torch import nn
from torch.nn.parallel import parallel_apply

import typing
from typing import List, Tuple, Optional


class ShadelessShader(nn.Module):
    def __init__(self, blend_params=None, device="cpu"):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()
            
    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, self.blend_params)
        return images, fragments.pix_to_face # (N, H, W, 3) RGBA image

class ReturnFragmentsShader(nn.Module):
    def __init__(self, blend_params=None, device="cpu"):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()
            
    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        return fragments # (N, H, W, 3) RGBA image
    
def get_fragments(cameras: List[CamerasBase],
                 mesh: Meshes,
                 image_size: int,
                 device: Optional[str]=None):

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(image_size=image_size,
                                        blur_radius=0.0,
                                        faces_per_pixel=1)
    renderers = [MeshRenderer(
                    rasterizer=MeshRasterizer(
                        cameras=camera, 
                        raster_settings=raster_settings
                    ),
                    shader=ReturnFragmentsShader( device=device, )
                ) for camera in cameras]

    # this could be serial to trade off speed for less memory consumption
    fragments_all = parallel_apply(renderers, [mesh] * len(renderers), devices=[device] * len(renderers))
    return fragments_all


# def aggregate_labels(fragments_all, logits_all, scaling=torch.sqrt, use_sparse_optimizations=True):
#     '''
#         fragments_all
#         logits_all
#     '''
#     # type: (List[Tensor],List[Tensor]) -> Tensor
#     # TODO: This will be very memory-inefficient for big meshes with a lot of faces.
#     # We can fix this by creating a mapping idx -> face_idx to make this dense again. 
#     # TODO: This only works for a single image. Need to make it work for batches. 

#     device = fragments_all[0].device
#     # only for a single image
#     faces_per_image = torch.tensor([int(f.max()) for f in fragments_all])
#     n_channels = logits_all[0].shape[-1]

#     assert logits_all.shape[1] == logits_all.shape[2], f"predictions should be shape B x W x H x C, not {logits_all.shape}. soft_combine only handles square predictions."

#     if use_sparse_optimizations:
#         all_faces = torch.cat([fragments.unique() for fragments in fragments_all]).unique()
#         face_to_compacted_idx = torch.zeros(int(all_faces.max() + 2), dtype=torch.int64, device=device)
#         face_to_compacted_idx[all_faces] = torch.arange(0, len(all_faces)).to(face_to_compacted_idx.device)
#         faces_compacted = torch.zeros((len(all_faces), n_channels), device=device)
#     else:
#         faces_compacted = torch.zeros((int(faces_per_image.max() + 2), n_channels), device=device)    
#         face_to_compacted_idx = None

#     # Aggregate labels from each input prediction
# #     for fragments, logits in zip(fragments_all, logits_all):
#     def add_count(fragments, logits):
#         new_faces_compacted = torch.zeros_like(faces_compacted)
#         unique_faces, counts = fragments.detach().unique(return_counts=True)

#         # Do voting based on reprojection 
#         flat_frags = ein.rearrange(fragments, 'b h w c -> (c b h w)')
#         if use_sparse_optimizations:
#             flat_frags = face_to_compacted_idx[flat_frags]
        
#         flat_logits = ein.rearrange(logits, 'b h w c -> (c b h w)')
    
#         flat_frags_repeated = torch.cat(n_channels*[flat_frags])        
#         flat_idxs = torch.repeat_interleave(torch.arange(0, n_channels), 
#                                             repeats=flat_frags.numel(),
#                                             dim=0)

#         new_faces_compacted.index_put_((flat_frags_repeated, flat_idxs),
#                              flat_logits,
#                              accumulate=True)
        
#         scaled_counts = scaling(counts.view(-1, 1).float())

#         # Normalize and add
#         if use_sparse_optimizations:
#             faces_compacted[face_to_compacted_idx[unique_faces]] += (new_faces_compacted[face_to_compacted_idx[unique_faces]] / scaled_counts)
#             #  faces_compacted += (new_faces_compacted / scaled_counts)  # might be faster if high utilization
#         else:
#             faces_compacted[unique_faces] += (new_faces_compacted[unique_faces] / scaled_counts)
    
#     # Could actually just do: add_count(fragments_all, logits_all)
#     # Turns out this way is about 10% faster, and can use less memory.
#     logits_all = logits_all.unsqueeze(1)
#     fragments_all = fragments_all.unsqueeze(1)
#     parallel_apply([add_count] * len(fragments_all),
#                    list(zip(fragments_all, logits_all)),
#                    devices=[device] * len(fragments_all))


#     return faces_compacted, face_to_compacted_idx


def aggregate_labels(fragments_all, 
                     logits_all, 
                     scaling=torch.sqrt, 
                     use_sparse_optimizations=True, 
                     return_mask=False, 
                     parallel_aggregate=False, 
                     return_std=False):
    '''
        fragments_all
        logits_all
    '''
    # type: (List[Tensor],List[Tensor]) -> Tensor
    # TODO: This will be very memory-inefficient for big meshes with a lot of faces.
    # We can fix this by creating a mapping idx -> face_idx to make this dense again.
    # TODO: This only works for a single image. Need to make it work for batches.

    device = fragments_all[0].device
    # only for a single image
    faces_per_image = torch.tensor([int(f.max()) for f in fragments_all])
    n_channels = logits_all[0].shape[-1]

    assert logits_all.shape[1] == logits_all.shape[
        2], f"predictions should be shape B x W x H x C, not {logits_all.shape}. soft_combine only handles square predictions."

    if use_sparse_optimizations:
        all_faces = torch.cat([fragments.unique() for fragments in fragments_all]).unique()
        face_to_compacted_idx = torch.zeros(int(all_faces.max() + 2), dtype=torch.int64, device=device)
        face_to_compacted_idx[all_faces] = torch.arange(0, len(all_faces)).to(face_to_compacted_idx.device)
        faces_compacted = torch.zeros((len(all_faces), n_channels), device=device)
        counts_compacted = torch.zeros((len(all_faces)), device=device)
        view_counts = torch.zeros((len(all_faces)), device=device)
        if return_std:
            sum_x2 = torch.zeros((len(all_faces), n_channels), device=device)
    else:
        faces_compacted = torch.zeros((int(faces_per_image.max() + 2), n_channels), device=device)
        face_to_compacted_idx = None
        counts_compacted = torch.zeros((int(faces_per_image.max() + 2)), device=device)
        view_counts = torch.zeros((int(faces_per_image.max() + 2)), device=device)
        if return_std:
            sum_x2 = torch.zeros((int(faces_per_image.max() + 2), n_channels), device=device)

    # Aggregate labels from each input prediction
    #     for fragments, logits in zip(fragments_all, logits_all):
    def add_count(fragments, logits):
        new_faces_compacted = torch.zeros_like(faces_compacted)
        unique_faces, counts = fragments.detach().unique(return_counts=True)

        # Do voting based on reprojection
        flat_frags = ein.rearrange(fragments, 'b h w c -> (c b h w)')
        if use_sparse_optimizations:
            flat_frags = face_to_compacted_idx[flat_frags]

        flat_logits = ein.rearrange(logits, 'b h w c -> (c b h w)')

        flat_frags_repeated = torch.cat(n_channels * [flat_frags])
        flat_idxs = torch.repeat_interleave(torch.arange(0, n_channels),
                                            repeats=flat_frags.numel(),
                                            dim=0)

        new_faces_compacted.index_put_((flat_frags_repeated, flat_idxs),
                                       flat_logits,
                                       accumulate=True)

        for f in fragments.detach():
            f_unique = face_to_compacted_idx[f.unique()] if use_sparse_optimizations else f.unique()
            view_counts[f_unique] += 1

        scaled_counts = scaling(counts.float())

        # Normalize and add
        if use_sparse_optimizations:
            faces_compacted[face_to_compacted_idx[unique_faces]] += new_faces_compacted[face_to_compacted_idx[unique_faces]]
            counts_compacted[face_to_compacted_idx[unique_faces]] += scaled_counts
            if return_std:
                sum_x2[face_to_compacted_idx[unique_faces]] += new_faces_compacted[face_to_compacted_idx[unique_faces]]**2
        else:
            faces_compacted[unique_faces] += new_faces_compacted[unique_faces]
            counts_compacted[unique_faces] += scaled_counts
            if return_std:
                sum_x2[unique_faces] += new_faces_compacted[unique_faces]**2

    if parallel_aggregate:
        # Turns out this way is about 10% faster, and can use less memory. ( <- On COCO on 2000 series card? )
        logits_all = logits_all.unsqueeze(1)
        fragments_all = fragments_all.unsqueeze(1)
        parallel_apply([add_count] * len(fragments_all),
                    list(zip(fragments_all, logits_all)),
                    devices=[device] * len(fragments_all))
    else:
        # Using a V100, on depth and normals tasks, this is much faster
        add_count(fragments_all, logits_all)

    # Normalize by total scaled counts
    faces_compacted[counts_compacted != 0] /= counts_compacted[counts_compacted != 0].view(-1,1)
    
    if return_std:
        sum_x2[counts_compacted != 0] /= counts_compacted[counts_compacted != 0].view(-1,1)
        std = torch.sqrt(sum_x2 - faces_compacted**2)
        
    result = {}
    result['faces_compacted'] = faces_compacted
    result['face_to_compacted_idx'] = face_to_compacted_idx
    if return_mask:
        mask_valid = view_counts > 1.0
        result['mask_valid'] = mask_valid
    if return_std:
        result['std'] = std
    return result

        
def aggregate_3d(cameras: List[CamerasBase],
                 predictions: List[torch.Tensor],
                 mesh: Meshes,
                 image_size: int,
                 device: Optional[str]=None,
                 scaling_fn=None,
                 return_masks=False,
                 return_face_preds=False,
                 return_std=False,
                 use_sparse_optimizations=True,
                 parallel_aggregate=False,
                 fragments_all=None) -> List[torch.Tensor]:
    '''
        Args:
            cameras: A list of Cameras for which we will render target views of the given mesh. 
            predictions:  A list of B x W x H x C predictions corresponding to the given cameras. 
                These will be used to determine a 'consistent' labeling of the mesh.
            image_size: How big to render out the target predictions. Can be different than input predictions. 
            mesh: A 'Meshes' object.
                TODO: I think that this is supposed to be a batch.
            device: On which device to do the processing. 
            scaling_fn: A function that, if n_pixels in a prediction corresponding to a specific 
                mesh face, scaling_fn(n_pixels) determines the weighting. Specifically:
                >>>  prediction_contribution = sum(predictions_for_face) / scaling(n_pixels)
            return_masks: 
            return_face_preds: 
            return_std: 
            use_sparse_optimizations: Turn off for debugging
            parallel_aggregate: 
            fragments_all: 
    
        Returns:
            target_predictions: A list of 'consistent' predictions, rendered according to the 
                viewpoints specified in 'cameras'
    '''
    # Get pix -> face correspondences for each prediction
    if fragments_all is None:
        fragments = get_fragments(cameras, mesh, image_size, device)
        fragments_all = torch.cat([f.pix_to_face for f in fragments], dim=0)
    
    if scaling_fn is None:
        scaling_fn = lambda x: x
    
    # Aggregate labels in 3d
    result_3d_agg = aggregate_labels(
        fragments_all,
        predictions,
        scaling=scaling_fn,
        use_sparse_optimizations=use_sparse_optimizations,
        return_mask=True,
        parallel_aggregate=parallel_aggregate,
        return_std=return_std
    )
    face_labels, face_to_idx = result_3d_agg['faces_compacted'], result_3d_agg['face_to_compacted_idx']
    
    if return_masks:
        face_mask_valid = result_3d_agg['mask_valid']
        
    if return_std:
        face_std = result_3d_agg['std']
    
    # Reproject those labels back onto initial images
    if use_sparse_optimizations:
        target_predictions = [face_labels[face_to_idx[f.flatten()]].reshape(p.shape) for f, p in zip(fragments_all, predictions)]
        if return_masks:
            target_masks = [face_mask_valid[face_to_idx[f.flatten()]].reshape(p.shape[0], p.shape[1], 1) for f, p in 
                            zip(fragments_all, predictions)]
        if return_std:
            target_std = [face_std[face_to_idx[f.flatten()]].reshape(p.shape) for f, p in zip(fragments_all, predictions)]
    else: 
        target_predictions = [face_labels[f.flatten()].reshape(p.shape) for f, p in zip(fragments_all, predictions)]
        if return_masks:
            target_masks = [face_mask_valid[f.flatten()].reshape(p.shape[0], p.shape[1], 1) for f, p in
                            zip(fragments_all, predictions)]
        if return_std:
            target_std = [face_std[f.flatten()].reshape(p.shape) for f, p in zip(fragments_all, predictions)]
            
    result = {}
    result['target_predictions'] = torch.stack(target_predictions)
    if return_masks:
        target_masks = torch.stack(target_masks)
        result['mask_valid'] = (fragments_all != -1) & target_masks
        result['face_mask_valid'] = face_mask_valid
    if return_face_preds:
        result['face_preds'] = face_labels
    if return_std:
        result['std'] = torch.stack(target_std)
    return result


def get_pixel_coordinates(cameras: List[CamerasBase],
                          predictions: List[torch.Tensor],
                          building:str,
                          point:int,
                          device: Optional[str] = None,
                          max_distance: int = 8000.0 / 512.0,
                          eps: int = 1e-4):
    pix_to_coordinates = []

    for v, camera in enumerate(cameras):
        pred = predictions[v]
        H, W, _ = pred.shape
        ndc_points = torch.zeros([H * W, 3], device=device)
        # Convert screen coordinates to ndc coordinates :
        # screen_x = (W - 1) / 2 * (1 - ndc_x)
        # screen_y = (H - 1) / 2 ( (1 - ndc_y)
        ndc_x = torch.cat(H * [1 - 2 * torch.arange(W, dtype=torch.float64) / (W - 1)]).to(device)
        ndc_y = (1 - 2 * torch.arange(H, dtype=torch.float64) / (H - 1)).repeat_interleave(W).to(device)
        # Each point is (ndc_x, ndc_y, depth) and we should have cam.znear < depth < cam.zfar
        ndc_points[:, 0] = ndc_x
        ndc_points[:, 1] = ndc_y
        ndc_points[:, 2] = max_distance * ein.rearrange(pred.squeeze(2), 'h w -> (h w)') + eps

        # Convert (ndc_x, ndc_y, depth) to world coordinates (x, y, z)
        world_points = camera.unproject_points(ndc_points, world_coordinates=True)
        pix_to_coordinates.append(ein.rearrange(world_points, '(h w) c -> h w c', h=H))

    #plot(pix_to_coordinates, '{}_point_{}_non_consistent.png'.format(building, point))

    pix_to_coordinates = torch.stack(pix_to_coordinates).to(device)
    return pix_to_coordinates


def plot(pix_to_coordinates, filename):
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = ['salmon', 'yellowgreen', 'steelblue', 'crimson']
    for c in range(len(pix_to_coordinates)):
        temp = ein.rearrange(pix_to_coordinates[c], 'h w c -> (h w) c').detach().cpu().numpy()
        ax.scatter(temp[:, 0],
                   temp[:, 1],
                   zs=temp[:, 2],
                   s=2,
                   c=colors[c])

    plt.savefig(filename)
