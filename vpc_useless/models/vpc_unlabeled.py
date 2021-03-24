# -*- coding: utf-8 -*-
import einops as ein
import torch
from torch import nn
from torch.nn import functional as F
from   typing import Optional, List, Callable, Union, Dict, Any

from detectron2.modeling.meta_arch.semantic_seg import SemSegFPNHead
from detectron2.modeling.meta_arch.panoptic_fpn import combine_semantic_and_instance_outputs
from detectron2.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from detectron2.modeling import (
    PanopticFPN,
    META_ARCH_REGISTRY,
    SEM_SEG_HEADS_REGISTRY,
    PROPOSAL_GENERATOR_REGISTRY,
)
from detectron2.structures import ImageList, Instances


from vpc.consistency.taskonomy_mesh import create_taskonomy_camera
from vpc.consistency.viewpoint_aggregation import aggregate_3d
from vpc.data.consistency import CONSISTENCY_MESH_REGISTRY
    

__all__ = ["VPC", "VPCOnSemsegOnly", "SemSegFPNHeadThatReturnsPredictionsAndLosses", "RPNWithoutAssert"]



def dense_softmax_cross_entropy_loss(inputs, targets, weights=None, reduction='mean'):  # these should be logits  (batch_size, n_class)
    batch_size = targets.shape[0]
    loss = -1. * torch.softmax(targets, dim=1) * F.log_softmax(inputs, dim=1) 
    if weights is not None:
        loss = loss * weights
    if reduction == 'mean':
        loss = torch.mean(loss) #/ batch_size
    elif reduction == 'sum':
        loss = torch.sum(loss)
    elif reduction == None:
        loss = loss
    else:
        raise NotImplementedError(f'Unknown reduction {reduction}')
    return loss


@META_ARCH_REGISTRY.register()
class VPC(PanopticFPN):
    """
    Implement the paper :paper:`PanopticFPN`.
    """

    
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.consistency_label_size = cfg.SOLVER.CONSISTENCY_TARGET_SIZE
        self.consistency_loss_weight = cfg.SOLVER.CONSISTENCY_LOSS_WEIGHT
        self.mask_margin = cfg.MODEL.CONSISTENCY.MASK_MARGIN

    def _pred(self, batched_inputs, is_consistency_batch=False, exit_on_semseg=False):
        """
        Args:
            A list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * "image": Tensor, image in (C, H, W) format.
                * "instances": Instances
                * "sem_seg": semantic segmentation ground truth.
                * Other information that's included in the original dicts, such as:
                  "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
#         if is_consistency_batch:
#             self.proposal_generator.training = False
#             self.roi_heads.training = False

#         pdb.set_trace()
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "sem_seg" in batched_inputs[0]:
            gt_sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
            gt_sem_seg = ImageList.from_tensors(
                gt_sem_seg, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
        else:
            gt_sem_seg = None
        sem_seg_results, sem_seg_losses = self.sem_seg_head(features, gt_sem_seg)

        if exit_on_semseg:
            return {
                'images': images,
                'sem_seg_results': sem_seg_results,
                'sem_seg_losses': sem_seg_losses,
            }


        
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        if self.proposal_generator:
            # Maybe throw in an option to use no_grad for proposals?
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        elif "proposals" in batched_inputs[0]:
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
    
        detector_results, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances
        )

#         if is_consistency_batch:
#             self.proposal_generator.training = self.training
#             self.roi_heads.training = self.training
        return {
            'images': images,
            'gt_instances': gt_instances,
            'gt_sem_seg': gt_sem_seg,
            'proposals': proposals,
            'proposal_losses': proposal_losses,
            'detector_results': detector_results,
            'detector_losses': detector_losses,
            'sem_seg_results': sem_seg_results,
            'sem_seg_losses': sem_seg_losses,
            
        }
    
    def process_results(self, sem_seg_results, detector_results, batched_inputs, images, **kwargs):
        if detector_results is None: 
            detector_results = [None] * len(batched_inputs)
        processed_results = []
        for sem_seg_result, detector_result, input_per_image, image_size in zip(
            sem_seg_results, detector_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            sem_seg_r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
            
            if not self.training:
                detector_r = detector_postprocess(detector_result, height, width)

                processed_results.append({"sem_seg": sem_seg_r, "instances": detector_r})

                if self.combine_on:
                    panoptic_r = combine_semantic_and_instance_outputs(
                        detector_r,
                        sem_seg_r.argmax(dim=0),
                        self.combine_overlap_threshold,
                        self.combine_stuff_area_limit,
                        self.combine_instances_confidence_threshold,
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r
        return processed_results


    def losses(self, detector_losses, sem_seg_losses, proposal_losses, consistency_losses, **kwargs):
        losses = {}
        losses.update(sem_seg_losses)
        losses.update({k: v * self.instance_loss_weight for k, v in detector_losses.items()})
        losses.update(proposal_losses)
        losses.update(consistency_losses)
        return losses


    def _consistency(self, consistency_batch, scaling_fn=lambda x: x):
        # Predictions for consistency
        preds = self._pred(
            consistency_batch,
            is_consistency_batch=True
        )        
    
        device = preds['sem_seg_results'].device
        
        # Set up unprojection
        cameras = [create_taskonomy_camera(point['point_info'], device=device) for point in consistency_batch]

        original_stacked_preds = preds['sem_seg_results']
        stacked_preds = F.interpolate(
            original_stacked_preds,
            (self.consistency_label_size, self.consistency_label_size),
            mode='nearest'
        )
        stacked_preds_for_agg3d = ein.rearrange(stacked_preds, "b c w h -> b w h c")

        meshes = [point['point_info']['building'] for point in consistency_batch]
        mesh = meshes[0]
        mesh = CONSISTENCY_MESH_REGISTRY.get(mesh).to(device)
        assert len(set(meshes)) == 1
        consistency_targets, masks = aggregate_3d(
                            cameras,
                            stacked_preds_for_agg3d,
                            mesh=mesh,
                            image_size=self.consistency_label_size,
                            return_masks=True,
                            scaling_fn=scaling_fn,
        )

        
        # recombine
        consistency_targets = torch.stack(consistency_targets)
        consistency_targets = ein.rearrange(consistency_targets, "b w h c -> b c w h")

        margin = self.mask_margin
        masks_inv = ein.rearrange(~masks, 'b w h c -> b c w h')
        masks = F.max_pool2d(masks_inv.float(), 2 * margin + 1, padding=margin, stride=1) == 0
        
        loss = dense_softmax_cross_entropy_loss(
            stacked_preds,
            consistency_targets,
            weights=masks
        )
    


        return {
            'consistency_results': preds,
            'consistency_targets': consistency_targets,
            'consistency_masks': masks,
            'consistency_losses': {'consistency_loss': loss * self.consistency_loss_weight }
        }
    

        
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a tuple of batch labeled points, and then consistency batch
            Both batches are a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * "image": Tensor, image in (C, H, W) format.
                * "instances": Instances
                * "sem_seg": semantic segmentation ground truth.
                * Other information that's included in the original dicts, such as:
                  "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:
                * "instances": see :meth:`GeneralizedRCNN.forward` for its format.
                * "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
                * "panoptic_seg": available when `PANOPTIC_FPN.COMBINE.ENABLED`.
                  See the return value of
                  :func:`combine_semantic_and_instance_outputs` for its format.
        """
        
        
        """
            for stuff w/o labels, regenerate labels
        """
        if self.training:
            batched_inputs, consistency_inputs = batched_inputs

        coco_preds = self._pred(
                    batched_inputs,
                    is_consistency_batch=False
        )
            
        if self.training:
            consistency_results = self._consistency(consistency_inputs)
           
            return self.losses(
                detector_losses=coco_preds['detector_losses'],
                sem_seg_losses=coco_preds['sem_seg_losses'],
                proposal_losses=coco_preds['proposal_losses'],
                consistency_losses=consistency_results['consistency_losses'],
            )
        
       
        processed_results = self.process_results(
            batched_inputs=batched_inputs,
            **coco_preds
        )
        coco_preds['processed_results'] = processed_results
        return coco_preds['processed_results']

    
    

@META_ARCH_REGISTRY.register()
class VPCOnSemsegOnly(VPC):
    def _pred(self, batched_inputs, is_consistency_batch=False, exit_on_semseg=False):
        return super()._pred(batched_inputs, is_consistency_batch, exit_on_semseg=is_consistency_batch)



    
    



@SEM_SEG_HEADS_REGISTRY.register()
class SemSegFPNHeadThatReturnsPredictionsAndLosses(SemSegFPNHead):
    def forward(self, features, targets=None):
        """
        Returns:
            This was what it returned: 
                In training, returns (None, dict of losses)
                In inference, returns (CxHxW logits, {})
            That's stupid. Just return 
                (CxHxW logits, dict of losses)
        """
        x = self.layers(features)
        losses = {}
        if targets is not None:
            losses = self.losses(x, targets) # For some reason the interpolation here also happens in the loss. Whatever, just do it in two places.
        preds = F.interpolate(
                x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        return preds, losses
    
    
    

@PROPOSAL_GENERATOR_REGISTRY.register()
class RPNWithoutAssert(PROPOSAL_GENERATOR_REGISTRY.get("RPN")):
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
    ):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.
        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            .float()  # ensure fp32 for decoding precision
            for x in pred_anchor_deltas
        ]

        if self.training:
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
        else:
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        return proposals, losses