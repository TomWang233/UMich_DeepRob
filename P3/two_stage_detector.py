import math
from typing import Dict, List, Optional, Tuple

import torch
import torchvision
from p3_helper import *
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction


# Short hand type notation:
TensorDict = Dict[str, torch.Tensor]


def hello_two_stage_detector():
    print("Hello from two_stage_detector.py!")
    
    
    
class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(pretrained=True)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        #                                                                    #
        # Create THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #
        # Then create THREE "output" 3x3 conv layers to transform the merged #
        # FPN features to output (p3, p4, p5) features.                      #
        # All conv layers must have stride=1 and padding such that features  #
        # do not get downsampled due to 3x3 convs.                           #
        #                                                                    #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these layers.                         #
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        self.fpn_params = nn.ModuleDict()

        # Replace "pass" statement with your code
        for level_name, feature_shape in dummy_out_shapes:
            self.fpn_params['conv1_' + level_name] = nn.Conv2d(feature_shape[1], self.out_channels, 1)
            self.fpn_params['conv3_' + level_name] = nn.Conv2d(self.out_channels, self.out_channels, 3, padding='same')
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using RegNet features  #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################

        # Replace "pass" statement with your code
        scale_3_4 = self.fpn_strides['p4'] / self.fpn_strides['p3']
        scale_4_5 = self.fpn_strides['p5'] / self.fpn_strides['p4']
        p3 = self.fpn_params['conv1_c3'](backbone_feats['c3'])
        p4 = self.fpn_params['conv1_c4'](backbone_feats['c4'])
        p5 = self.fpn_params['conv1_c5'](backbone_feats['c5'])
        p4 = p4 + F.interpolate(p5, scale_factor=scale_4_5)
        p3 = p3 + F.interpolate(p4, scale_factor=scale_3_4)
        fpn_feats['p3'] = self.fpn_params['conv3_c3'](p3)
        fpn_feats['p4'] = self.fpn_params['conv3_c4'](p4)
        fpn_feats['p5'] = self.fpn_params['conv3_c5'](p5)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats


class RPNPredictionNetwork(nn.Module):
    """
    RPN prediction network that accepts FPN feature maps from different levels
    and makes two predictions for every anchor: objectness and box deltas.

    Faster R-CNN typically uses (p2, p3, p4, p5) feature maps. We will exclude
    p2 for have a small enough model for Colab.

    Conceptually this module is quite similar to `FCOSPredictionNetwork`.
    """

    def __init__(
        self, in_channels: int, stem_channels: List[int], num_anchors: int = 3
    ):
        """
        Args:
            in_channels: Number of channels in input feature maps. This value
                is same as the output channels of FPN.
            stem_channels: List of integers giving the number of output channels
                in each convolution layer of stem layers.
            num_anchors: Number of anchor boxes assumed per location (say, `A`).
                Faster R-CNN without an FPN uses `A = 9`, anchors with three
                different sizes and aspect ratios. With FPN, it is more common
                to have a fixed size dependent on the stride of FPN level, hence
                `A = 3` is default - with three aspect ratios.
        """
        super().__init__()

        self.num_anchors = num_anchors
        ######################################################################
        # TODO: Create a stem of alternating 3x3 convolution layers and RELU
        # activation modules. RPN shares this stem for objectness and box
        # regression (unlike FCOS, that uses separate stems).
        #
        # Use `in_channels` and `stem_channels` for creating these layers, the
        # docstring above tells you what they mean. Initialize weights of each
        # conv layer from a normal distribution with mean = 0 and std dev = 0.01
        # and all biases with zero. Use conv stride = 1 and zero padding such
        # that size of input features remains same: remember we need predictions
        # at every location in feature map, we shouldn't "lose" any locations.
        ######################################################################
        # Fill this list. It is okay to use your implementation from
        # `FCOSPredictionNetwork` for this code block.
        stem_rpn = []
        # Replace "pass" statement with your code
        c = [in_channels, *stem_channels]
        for i in range(len(stem_channels)):
            stem_rpn.append(nn.Conv2d(c[i], c[i + 1], 3, padding='same'))
            nn.init.normal_(stem_rpn[2 * i].weight, 0, 0.01)
            nn.init.constant_(stem_rpn[2 * i].bias, 0)
            stem_rpn.append(nn.ReLU())
        # Wrap the layers defined by student into a `nn.Sequential` module:
        self.stem_rpn = nn.Sequential(*stem_rpn)
        ######################################################################
        # TODO: Create TWO 1x1 conv layers for individually to predict
        # objectness and box deltas for every anchor, at every location.
        #
        # Objectness is obtained by applying sigmoid to its logits. However,
        # DO NOT initialize a sigmoid module here. PyTorch loss functions have
        # numerically stable implementations with logits.
        ######################################################################

        # Replace these lines with your code, keep variable names unchanged.
        self.pred_obj = None  # Objectness conv
        self.pred_box = None  # Box regression conv

        # Replace "pass" statement with your code
        self.pred_obj = nn.Conv2d(c[-1], num_anchors, 1, padding='same')
        self.pred_box = nn.Conv2d(c[-1], 4 * num_anchors, 1, padding='same')
        nn.init.normal_(self.pred_obj.weight, 0, 0.01)
        nn.init.constant_(self.pred_obj.bias, 0)
        nn.init.normal_(self.pred_box.weight, 0, 0.01)
        nn.init.constant_(self.pred_box.bias, 0)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        """
        Accept FPN feature maps and predict desired quantities for every anchor
        at every location. Format the output tensors such that feature height,
        width, and number of anchors are collapsed into a single dimension (see
        description below in "Returns" section) this is convenient for computing
        loss and perforning inference.

        Args:
            feats_per_fpn_level: Features from FPN, keys {"p3", "p4", "p5"}.
                Each tensor will have shape `(batch_size, fpn_channels, H, W)`.

        Returns:
            List of dictionaries, each having keys {"p3", "p4", "p5"}:
            1. Objectness logits:     `(batch_size, H * W * num_anchors)`
            2. Box regression deltas: `(batch_size, H * W * num_anchors, 4)`
        """

        ######################################################################
        # TODO: Iterate over every FPN feature map and obtain predictions using
        # the layers defined above. DO NOT apply sigmoid to objectness logits.
        ######################################################################
        # Fill these with keys: {"p3", "p4", "p5"}, same as input dictionary.
        object_logits = {}
        boxreg_deltas = {}

        # Replace "pass" statement with your code
        for key, feat in feats_per_fpn_level.items():
            object_logits[key] = self.pred_obj(self.stem_rpn(feat)).flatten(1)
            boxreg_deltas[key] = self.pred_box(self.stem_rpn(feat))
            batch_size, _, H, W = boxreg_deltas[key].shape
            boxreg_deltas[key] = boxreg_deltas[key].reshape(batch_size, 4, -1, H, W).flatten(2).transpose(1, 2)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        return [object_logits, boxreg_deltas]


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################
        # Replace "pass" statement with your code
        h = torch.arange(feat_shape[2], device=device) + 0.5
        w = torch.arange(feat_shape[3], device=device) + 0.5
        location_coords[level_name] = torch.cartesian_prod(h, w) * level_stride
        location_coords[level_name] = location_coords[level_name].reshape(feat_shape[2], feat_shape[3], 2).transpose(0,1).reshape(-1, 2)
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords


@torch.no_grad()
def generate_fpn_anchors(
    locations_per_fpn_level: TensorDict,
    strides_per_fpn_level: Dict[str, int],
    stride_scale: int,
    aspect_ratios: List[float] = [0.5, 1.0, 2.0],
):
    """
    Generate multiple anchor boxes at every location of FPN level. Anchor boxes
    should be in XYXY format and they should be centered at the given locations.

    Args:
        locations_per_fpn_level: Centers at different levels of FPN (p3, p4, p5),
            that are already projected to absolute co-ordinates in input image
            dimension. Dictionary of three keys: (p3, p4, p5) giving tensors of
            shape `(H * W, 2)` where H, W is the size of FPN feature map.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `common.py` for more details.
        stride_scale: Size of square anchor at every FPN levels will be
            `(this value) * (FPN level stride)`. Default is 4, which will make
            anchor boxes of size (32x32), (64x64), (128x128) for FPN levels
            p3, p4, and p5 respectively.
        aspect_ratios: Anchor aspect ratios to consider at every location. We
            consider anchor area to be `(stride_scale * FPN level stride) ** 2`
            and set new width and height of anchors at every location:
                new_width = sqrt(area / aspect ratio)
                new_height = area / new_width

    Returns:
        TensorDict
            Dictionary with same keys as `locations_per_fpn_level` and values as
            tensors of shape `(HWA, 4)` giving anchors for all locations
            per FPN level, each location having `A = len(aspect_ratios)` anchors.
            All anchors are in XYXY format and their centers align with locations.
    """

    # Set these to `(N, A, 4)` Tensors giving anchor boxes in XYXY format.
    anchors_per_fpn_level = {
        level_name: None for level_name, _ in locations_per_fpn_level.items()
    }

    for level_name, locations in locations_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        # List of `A = len(aspect_ratios)` anchor boxes.
        anchor_boxes = []
        for aspect_ratio in aspect_ratios:
            ##################################################################
            # TODO: Implement logic for anchor boxes below. Write vectorized
            # implementation to generate anchors for a single aspect ratio.
            # Fill `anchor_boxes` list above.
            #
            # Calculate resulting width and height of the anchor box as per
            # `stride_scale` and `aspect_ratios` definitions. Then shift the
            # locations to get top-left and bottom-right co-ordinates.
            ##################################################################
            # Replace "pass" statement with your code
            area = torch.tensor((stride_scale * level_stride) ** 2, device=locations.device)
            w = torch.sqrt(area / aspect_ratio)
            h = area / w
            anchor_boxes.append(torch.hstack((locations, locations)) + torch.tensor((-w / 2, -h / 2, w / 2, h / 2),device=locations.device))
            ##################################################################
            #                           END OF YOUR CODE                     #
            ##################################################################

        # shape: (A, H * W, 4)
        anchor_boxes = torch.stack(anchor_boxes)
        # Bring `H * W` first and collapse those dimensions.
        anchor_boxes = anchor_boxes.permute(1, 0, 2).contiguous().view(-1, 4)
        anchors_per_fpn_level[level_name] = anchor_boxes

    return anchors_per_fpn_level


@torch.no_grad()
def iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute intersection-over-union (IoU) between pairs of box tensors. Input
    box tensors must in XYXY format.

    Args:
        boxes1: Tensor of shape `(M, 4)` giving a set of box co-ordinates.
        boxes2: Tensor of shape `(N, 4)` giving another set of box co-ordinates.

    Returns:
        torch.Tensor
            Tensor of shape (M, N) with `iou[i, j]` giving IoU between i-th box
            in `boxes1` and j-th box in `boxes2`.
    """

    ##########################################################################
    # TODO: Implement the IoU function here.                                 #
    ##########################################################################
    # Replace "pass" statement with your code
    M, N = boxes1.shape[0], boxes2.shape[0]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]))
    x1 = torch.maximum(boxes1[:, 0].unsqueeze(1), boxes2[:, 0])
    y1 = torch.maximum(boxes1[:, 1].unsqueeze(1), boxes2[:, 1])
    x2 = torch.minimum(boxes1[:, 2].unsqueeze(1), boxes2[:, 2])
    y2 = torch.minimum(boxes1[:, 3].unsqueeze(1), boxes2[:, 3])
    invalid_idx = torch.logical_or(x1 > x2, y1 > y2)
    inter_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    ibr = inter_boxes.reshape(-1, 4)
    inter_areas = ((ibr[:, 2] - ibr[:, 0]) * (ibr[:, 3] - ibr[:, 1])).reshape(M, N)
    union_areas = area1.unsqueeze(-1) + area2 - inter_areas 
    iou = inter_areas / union_areas
    iou[invalid_idx] = 0
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return iou


@torch.no_grad()
def rcnn_match_anchors_to_gt(
    anchor_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thresholds: Tuple[float, float],
) -> TensorDict:
    """
    Match anchor boxes (or RPN proposals) with a set of GT boxes. Anchors having
    high IoU with any GT box are assigned "foreground" and matched with that box
    or vice-versa.

    NOTE: This function is NOT BATCHED. Call separately for GT boxes per image.

    Args:
        anchor_boxes: Anchor boxes (or RPN proposals). Dictionary of three keys
            a combined tensor of some shape `(N, 4)` where `N` are total anchors
            from all FPN levels, or a set of RPN proposals.
        gt_boxes: GT boxes of a single image, a batch of `(M, 5)` boxes with
            absolute co-ordinates and class ID `(x1, y1, x2, y2, C)`. In this
            codebase, this tensor is directly served by the dataloader.
        iou_thresholds: Tuple of (low, high) IoU thresholds, both in [0, 1]
            giving thresholds to assign foreground/background anchors.
    """

    # Filter empty GT boxes:
    gt_boxes = gt_boxes[gt_boxes[:, 4] != -1]

    # If no GT boxes are available, match all anchors to background and return.
    if len(gt_boxes) == 0:
        fake_boxes = torch.zeros_like(anchor_boxes) - 1
        fake_class = torch.zeros_like(anchor_boxes[:, [0]]) - 1
        return torch.cat([fake_boxes, fake_class], dim=1)

    # Match matrix => pairwise IoU of anchors (rows) and GT boxes (columns).
    # STUDENTS: This matching depends on your IoU implementation.
    match_matrix = iou(anchor_boxes, gt_boxes[:, :4])

    # Find matched ground-truth instance per anchor:
    match_quality, matched_idxs = match_matrix.max(dim=1)
    matched_gt_boxes = gt_boxes[matched_idxs]

    # Set boxes with low IoU threshold to background (-1).
    matched_gt_boxes[match_quality <= iou_thresholds[0]] = -1

    # Set remaining boxes to neutral (-1e8).
    neutral_idxs = (match_quality > iou_thresholds[0]) & (
        match_quality < iou_thresholds[1]
    )
    matched_gt_boxes[neutral_idxs, :] = -1e8
    return matched_gt_boxes


def rcnn_get_deltas_from_anchors(
    anchors: torch.Tensor, gt_boxes: torch.Tensor
) -> torch.Tensor:
    """
    Get box regression deltas that transform `anchors` to `gt_boxes`. These
    deltas will become GT targets for box regression. Unlike FCOS, the deltas
    are in `(dx, dy, dw, dh)` format that represent offsets to anchor centers
    and scaling factors for anchor size. Box regression is only supervised by
    foreground anchors. If GT boxes are "background/neutral", then deltas
    must be `(-1e8, -1e8, -1e8, -1e8)` (just some LARGE negative number).

    Follow lecture 12:
        https://deeprob.org/calendar/#lec-12

    Args:
        anchors: Tensor of shape `(N, 4)` giving anchors boxes in XYXY format.
        gt_boxes: Tensor of shape `(N, 4)` giving matching GT boxes.

    Returns:
        torch.Tensor
            Tensor of shape `(N, 4)` giving anchor deltas.
    """
    ##########################################################################
    # TODO: Implement the logic to get deltas.                               #
    # Remember to set the deltas of "background/neutral" GT boxes to -1e8    #
    ##########################################################################
    deltas = None
    # Replace "pass" statement with your code
    deltas = torch.zeros_like(anchors)
    idx = gt_boxes[:, 0] > -1
    a = anchors[idx]
    gt = gt_boxes[idx]
    px = (a[:, 0] + a[:, 2]) / 2
    py = (a[:, 1] + a[:, 3]) / 2
    pw = a[:, 2] - a[:, 0]
    ph = a[:, 3] - a[:, 1]
    bx = (gt[:, 0] + gt[:, 2]) / 2
    by = (gt[:, 1] + gt[:, 3]) / 2
    bw = gt[:, 2] - gt[:, 0]
    bh = gt[:, 3] - gt[:, 1]
    tx = (bx - px) / pw
    ty = (by - py) / ph
    tw = torch.log(bw / pw)
    th = torch.log(bh / ph)
    deltas[idx] = torch.vstack((tx, ty, tw, th)).T
    idx = torch.logical_not(idx)
    deltas[idx] = -1e8
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return deltas


def rcnn_apply_deltas_to_anchors(
    deltas: torch.Tensor, anchors: torch.Tensor
) -> torch.Tensor:
    """
    Implement the inverse of `rcnn_get_deltas_from_anchors` here.

    Args:
        deltas: Tensor of shape `(N, 4)` giving box regression deltas.
        anchors: Tensor of shape `(N, 4)` giving anchors to apply deltas on.

    Returns:
        torch.Tensor
            Same shape as deltas and locations, giving the resulting boxes in
            XYXY format.
    """

    # Clamp dw and dh such that they would transform a 8px box no larger than
    # 224px. This is necessary for numerical stability as we apply exponential.
    scale_clamp = math.log(224 / 8)
    deltas[:, 2] = torch.clamp(deltas[:, 2], max=scale_clamp)
    deltas[:, 3] = torch.clamp(deltas[:, 3], max=scale_clamp)

    ##########################################################################
    # TODO: Implement the transformation logic to get output boxes.          #
    ##########################################################################
    output_boxes = None
    # Replace "pass" statement with your code
    output_boxes = torch.zeros_like(anchors)
    px = (anchors[:, 0] + anchors[:, 2]) / 2
    py = (anchors[:, 1] + anchors[:, 3]) / 2
    pw = anchors[:, 2] - anchors[:, 0]
    ph = anchors[:, 3] - anchors[:, 1]
    bx = px + pw * deltas[:, 0]
    by = py + ph * deltas[:, 1]
    bw = pw * torch.exp(deltas[:, 2])
    bh = ph * torch.exp(deltas[:, 3])
    output_boxes[:, 0] = bx - bw / 2
    output_boxes[:, 1] = by - bh / 2
    output_boxes[:, 2] = bx + bw / 2
    output_boxes[:, 3] = by + bh / 2
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return output_boxes


@torch.no_grad()
def sample_rpn_training(
    gt_boxes: torch.Tensor, num_samples: int, fg_fraction: float
):
    """
    Return `num_samples` (or fewer, if not enough found) random pairs of anchors
    and GT boxes without exceeding `fg_fraction * num_samples` positives, and
    then try to fill the remaining slots with background anchors. We will ignore
    "neutral" anchors in this sampling as they are not used for training.

    Args:
        gt_boxes: Tensor of shape `(N, 5)` giving GT box co-ordinates that are
            already matched with some anchor boxes (with GT class label at last
            dimension). Label -1 means background and -1e8 means meutral.
        num_samples: Total anchor-GT pairs with label >= -1 to return.
        fg_fraction: The number of subsampled labels with values >= 0 is
            `min(num_foreground, int(fg_fraction * num_samples))`. In other
            words, if there are not enough fg, the sample is filled with
            (duplicate) bg.

    Returns:
        fg_idx, bg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or
            fewer. Use these to index anchors, GT boxes, and model predictions.
    """
    foreground = (gt_boxes[:, 4] >= 0).nonzero().squeeze(1)
    background = (gt_boxes[:, 4] == -1).nonzero().squeeze(1)

    # Protect against not enough foreground examples.
    num_fg = min(int(num_samples * fg_fraction), foreground.numel())
    num_bg = num_samples - num_fg

    # Randomly select positive and negative examples.
    perm1 = torch.randperm(foreground.numel(), device=foreground.device)[:num_fg]
    perm2 = torch.randperm(background.numel(), device=background.device)[:num_bg]

    fg_idx = foreground[perm1]
    bg_idx = background[perm2]
    return fg_idx, bg_idx


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    #############################################################################
    # TODO: Implement non-maximum suppression which iterates the following:     #
    #       1. Select the highest-scoring box among the remaining ones,         #
    #          which has not been chosen in this step before                    #
    #       2. Eliminate boxes with IoU > threshold                             #
    #       3. If any boxes remain, GOTO 1                                      #
    #       Your implementation should not depend on a specific device type;    #
    #       you can use the device of the input if necessary.                   #
    # HINT: You can refer to the torchvision library code:                      #
    # github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
    #############################################################################
    # Replace "pass" statement with your code
    DEVICE = boxes.device
    N = scores.shape[0]
    keep = torch.zeros(N, dtype=torch.long, device=DEVICE)
    num_kept_boxes = 0
    _, indices = torch.sort(scores, descending=True)
    areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    suppressed = torch.tensor([False] * N)
    for i in range(N):
        indi = indices[i]
        if suppressed[indi]:
            continue
        keep[num_kept_boxes] = indi
        num_kept_boxes += 1
        ix1, iy1, ix2, iy2 = boxes[indi]
        xx1 = torch.clamp(boxes[:, 0], min=ix1)
        yy1 = torch.clamp(boxes[:, 1], min=iy1)
        xx2 = torch.clamp(boxes[:, 2], max=ix2)
        yy2 = torch.clamp(boxes[:, 3], max=iy2)
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h
        iou = inter / (areas + areas[indi] - inter)
        suppressed[iou > iou_threshold] = True
    keep = keep[:num_kept_boxes]
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return keep


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


@torch.no_grad()
def reassign_proposals_to_fpn_levels(
    proposals_per_image: List[torch.Tensor],
    gt_boxes: Optional[torch.Tensor] = None,
    fpn_level_ids: List[int] = [3, 4, 5],
) -> Dict[str, List[torch.Tensor]]:
    """
    The first-stage in Faster R-CNN (RPN) gives a few proposals that are likely
    to contain any object. These proposals would have come from any FPN level -
    for example, they all maybe from level p5, and none from levels p3/p4 (= the
    image mostly has large objects and no small objects). In second stage, these
    proposals are used to extract image features (via RoI-align) and predict the
    class labels. But we do not know which level to use, due to two reasons:

        1. We did not keep track of which level each proposal came from.
        2. ... even if we did keep track, it may be possible that RPN deltas
           transformed a large anchor box from p5 to a tiny proposal (which could
           be more suitable for a lower FPN level).

    Hence, we re-assign proposals to different FPN levels according to sizes.
    Large proposals get assigned to higher FPN levels, and vice-versa.

    At start of training, RPN proposals may be low quality. It's possible that
    very few of these have high IoU with GT boxes. This may stall or de-stabilize
    training of second stage. This function also mixes GT boxes with RPN proposals
    to improve training. GT boxes are also assigned by their size.

    See Equation (1) in FPN paper (https://arxiv.org/abs/1612.03144).

    Args:
        proposals_per_image: List of proposals per image in batch. Same as the
            outputs from `RPN.forward()` method.
        gt_boxes: Tensor of shape `(B, M, 4 or 5)` giving GT boxes per image in
            batch (with or without GT class label, doesn't matter). These are
            not present during inference.
        fpn_levels: List of FPN level IDs. For this codebase this will always
            be [3, 4, 5] for levels (p3, p4, p5) -- we include this in input
            arguments to avoid any hard-coding in function body.

    Returns:
        Dict[str, List[torch.Tensor]]
            Dictionary with keys `{"p3", "p4", "p5"}` each containing a list
            of `B` (`batch_size`) tensors. The `i-th` element in this list will
            give proposals of `i-th` image, assigned to that FPN level. An image
            may not have any proposals for a particular FPN level, for which the
            tensor will be a tensor of shape `(0, 4)` -- PyTorch supports this!
    """

    # Make empty lists per FPN level to add assigned proposals for every image.
    proposals_per_fpn_level = {f"p{_id}": [] for _id in fpn_level_ids}

    # Usually 3 and 5.
    lowest_level_id, highest_level_id = min(fpn_level_ids), max(fpn_level_ids)

    for idx, _props in enumerate(proposals_per_image):

        # Mix ground-truth boxes for every example, per FPN level.
        if gt_boxes is not None:
            # Filter empty GT boxes and remove class label.
            _gtb = gt_boxes[idx]
            _props = torch.cat([_props, _gtb[_gtb[:, 4] != -1][:, :4]], dim=0)

        # Compute FPN level assignments for each GT box. This follows Equation (1)
        # of FPN paper (k0 = 4). `level_assn` has `(M, )` integers, one of {3,4,5}
        _areas = (_props[:, 2] - _props[:, 0]) * (_props[:, 3] - _props[:, 1])

        # Assigned FPN level ID for each proposal (an integer between lowest_level
        # and highest_level).
        level_assignments = torch.floor(4 + torch.log2(torch.sqrt(_areas) / 224))
        level_assignments = torch.clamp(
            level_assignments, min=lowest_level_id, max=highest_level_id
        )
        level_assignments = level_assignments.to(torch.int64)

        # Iterate over FPN level IDs and get proposals for each image, that are
        # assigned to that level.
        for _id in range(lowest_level_id, highest_level_id + 1):
            proposals_per_fpn_level[f"p{_id}"].append(
                # This tensor may have zero proposals, and that's okay.
                _props[level_assignments == _id]
            )

    return proposals_per_fpn_level


class RPN(nn.Module):
    """
    Region Proposal Network: First stage of Faster R-CNN detector.

    This class puts together everything you implemented so far. It accepts FPN
    features as input and uses `RPNPredictionNetwork` to predict objectness and
    box reg deltas. Computes proposal boxes for second stage (during both
    training and inference) and losses during training.
    """

    def __init__(
        self,
        fpn_channels: int,
        stem_channels: List[int],
        batch_size_per_image: int,
        anchor_stride_scale: int = 8,
        anchor_aspect_ratios: List[int] = [0.5, 1.0, 2.0],
        anchor_iou_thresholds: Tuple[int, int] = (0.3, 0.6),
        nms_thresh: float = 0.7,
        pre_nms_topk: int = 400,
        post_nms_topk: int = 100,
    ):
        """
        Args:
            batch_size_per_image: Anchors per image to sample for training.
            nms_thresh: IoU threshold for NMS - unlike FCOS, this is used
                during both, training and inference.
            pre_nms_topk: Number of top-K proposals to select before applying
                NMS, per FPN level. This helps in speeding up NMS.
            post_nms_topk: Number of top-K proposals to select after applying
                NMS, per FPN level. NMS is obviously going to be class-agnostic.

        Refer explanations of remaining args in the classes/functions above.
        """
        super().__init__()
        self.pred_net = RPNPredictionNetwork(
            fpn_channels, stem_channels, num_anchors=len(anchor_aspect_ratios)
        )
        # Record all input arguments:
        self.batch_size_per_image = batch_size_per_image
        self.anchor_stride_scale = anchor_stride_scale
        self.anchor_aspect_ratios = anchor_aspect_ratios
        self.anchor_iou_thresholds = anchor_iou_thresholds
        self.nms_thresh = nms_thresh
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk

    def forward(
        self,
        feats_per_fpn_level: TensorDict,
        strides_per_fpn_level: TensorDict,
        gt_boxes: Optional[torch.Tensor] = None,
    ):
        # Get batch size from FPN feats:
        num_images = feats_per_fpn_level["p3"].shape[0]

        ######################################################################
        # TODO: Implement the training forward pass. Follow these steps:
        #   1. Pass the FPN features per level to the RPN prediction network.
        #   2. Generate anchor boxes for all FPN levels.
        #
        # HINT: You have already implemented everything, just have to call the
        # appropriate functions.
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        pred_obj_logits, pred_boxreg_deltas, anchors_per_fpn_level = (
            None,
            None,
            None,
        )
        # Replace "pass" statement with your code
        pred_obj_logits, pred_boxreg_deltas = self.pred_net(feats_per_fpn_level)
        fpn_feats_shapes = {
            level: feat.shape for level, feat in feats_per_fpn_level.items()
        }
        DEVICE = list(feats_per_fpn_level.values())[0].device
        locations_per_fpn_level = get_fpn_location_coords(
            fpn_feats_shapes, strides_per_fpn_level, device=DEVICE
        )
        anchors_per_fpn_level = generate_fpn_anchors(
            locations_per_fpn_level,
            strides_per_fpn_level,
            self.anchor_stride_scale,
            self.anchor_aspect_ratios
        )
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # We will fill three values in this output dict - "proposals",
        # "loss_rpn_box" (training only), "loss_rpn_obj" (training only)
        output_dict = {}

        # Get image height and width according to feature sizes and strides.
        # We need these to clamp proposals (These should be (224, 224) but we
        # avoid hard-coding them).
        img_h = feats_per_fpn_level["p3"].shape[2] * strides_per_fpn_level["p3"]
        img_w = feats_per_fpn_level["p3"].shape[3] * strides_per_fpn_level["p3"]

        # STUDENT: Implement this method before moving forward with the rest
        # of this `forward` method.
        output_dict["proposals"] = self.predict_proposals(
            anchors_per_fpn_level,
            pred_obj_logits,
            pred_boxreg_deltas,
            (img_w, img_h),
        )
        # Return here during inference - loss computation not required.
        if not self.training:
            return output_dict

        # ... otherwise continue loss computation:
        ######################################################################
        # Match the generated anchors with provided GT boxes. This
        # function is not batched so you may use a for-loop, like FCOS.
        ######################################################################
        # Combine anchor boxes from all FPN levels - we do not need any
        # distinction of boxes across different levels (for training).
        anchor_boxes = self._cat_across_fpn_levels(anchors_per_fpn_level, dim=0)

        # Get matched GT boxes (list of B tensors, each of shape `(H*W*A, 5)`
        # giving matching GT boxes to anchor boxes). Fill this list:
        matched_gt_boxes = []
        # Replace "pass" statement with your code
        for i in range(num_images):
            matched_gt_boxes.append(rcnn_match_anchors_to_gt(
                anchor_boxes, gt_boxes[i], self.anchor_iou_thresholds
            ))
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Combine matched boxes from all images to a `(B, HWA, 5)` tensor.
        matched_gt_boxes = torch.stack(matched_gt_boxes, dim=0)

        # Combine predictions across all FPN levels.
        pred_obj_logits = self._cat_across_fpn_levels(pred_obj_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)

        if self.training:
            # Repeat anchor boxes `batch_size` times so there is a 1:1
            # correspondence with GT boxes.
            anchor_boxes = anchor_boxes.unsqueeze(0).repeat(num_images, 1, 1)
            anchor_boxes = anchor_boxes.contiguous().view(-1, 4)

            # Collapse `batch_size`, and `HWA` to a single dimension so we have
            # simple `(-1, 4 or 5)` tensors. This simplifies loss computation.
            matched_gt_boxes = matched_gt_boxes.view(-1, 5)
            pred_obj_logits = pred_obj_logits.view(-1)
            pred_boxreg_deltas = pred_boxreg_deltas.view(-1, 4)

            ##################################################################
            # TODO: Compute training losses. Follow three steps in order:
            #   1. Sample a few anchor boxes for training. Pass the variable
            #      `matched_gt_boxes` to `sample_rpn_training` function and
            #      use those indices to get subset of predictions and targets.
            #      RPN samples 50-50% foreground/background anchors, unless
            #      there aren't enough foreground anchors.
            #
            #   2. Compute GT targets for box regression (you have implemented
            #      the transformation function already).
            #
            #   3. Calculate objectness and box reg losses per sampled anchor.
            #      Remember to set box loss for "background" anchors to 0.
            ##################################################################
            # Feel free to delete this line: (but keep variable names same)
            loss_obj, loss_box = None, None
            # Replace "pass" statement with your code
            fg_idx, bg_idx = sample_rpn_training(
                matched_gt_boxes, self.batch_size_per_image * num_images, 0.5
            )
            gt_objectness = torch.hstack((
                matched_gt_boxes[fg_idx, -1], matched_gt_boxes[bg_idx, -1]
            ))
            fg_len = fg_idx.shape[0]
            gt_objectness[:fg_len] = 1
            gt_objectness[fg_len:] = 0
            fgbg_logits = torch.hstack((
                pred_obj_logits[fg_idx], pred_obj_logits[bg_idx]
            ))
            gt_deltas = rcnn_get_deltas_from_anchors(anchor_boxes, matched_gt_boxes)
            gt_deltas[bg_idx, :] = -1e8
            loss_obj = F.binary_cross_entropy_with_logits(
                fgbg_logits, gt_objectness, reduction="none"
            )
            loss_box = 0.25 * F.l1_loss(
                pred_boxreg_deltas, gt_deltas, reduction="none"
            )
            loss_box[gt_deltas == -1e8] *= 0.0
            ##################################################################
            #                         END OF YOUR CODE                       #
            ##################################################################

            # Sum losses and average by num(foreground + background) anchors.
            # In training code, we simply add these two and call `.backward()`
            total_batch_size = self.batch_size_per_image * num_images
            output_dict["loss_rpn_obj"] = loss_obj.sum() / total_batch_size
            output_dict["loss_rpn_box"] = loss_box.sum() / total_batch_size

        return output_dict

    
    @torch.no_grad()  # Don't track gradients in this function.
    def predict_proposals(
        self,
        anchors_per_fpn_level: Dict[str, torch.Tensor],
        pred_obj_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        image_size: Tuple[int, int],  # (width, height)
    ) -> List[torch.Tensor]:
        """
        Predict proposals for a batch of images for the second stage. Other
        input arguments are same as those computed in `forward` method. This
        method should not be called from anywhere except from inside `forward`.

        Returns:
            List[torch.Tensor]
                proposals_per_image: List of B (`batch_size`) tensors givine RPN
                proposals per image. These are boxes in XYXY format, that are
                most likely to contain *any* object. Each tensor in the list has
                shape `(N, 4)` where N could be variable for each image (maximum
                value `post_nms_topk`). These will be anchors for second stage.
        """

        # Gather RPN proposals *from all FPN levels* per image. This will be a
        # list of B (batch_size) tensors giving `(N, 4)` proposal boxes in XYXY
        # format (maximum value of N should be `post_nms_topk`).
        proposals_per_image = []

        # Get batch size to iterate over:
        batch_size = pred_obj_logits["p3"].shape[0]

        for _batch_idx in range(batch_size):

            # For each image in batch, iterate over FPN levels. Fill proposals
            # AND scores per image, per FPN level, in these:
            proposals_per_fpn_level_per_image = {
                level_name: None for level_name in anchors_per_fpn_level.keys()
            }
            scores_per_fpn_level_per_image = {
                level_name: None for level_name in anchors_per_fpn_level.keys()
            }

            for level_name in anchors_per_fpn_level.keys():

                # Get anchor boxes and predictions from a single level.
                level_anchors = anchors_per_fpn_level[level_name]

                # Predictions for a single image - shape: (HWA, ), (HWA, 4)
                level_obj_logits = pred_obj_logits[level_name][_batch_idx]
                level_boxreg_deltas = pred_boxreg_deltas[level_name][_batch_idx]

                ##############################################################
                # TODO: Perform the following steps in order:
                #   1. Transform the anchors to proposal boxes using predicted
                #      box deltas, clamp to image height and width.
                #   2. Sort all proposals by their predicted objectness, and
                #      retain `self.pre_nms_topk` proposals. This speeds up
                #      our NMS computation. HINT: `torch.topk`
                #   3. Apply NMS and add the filtered proposals and scores
                #      (logits, with or without sigmoid, doesn't matter) to
                #      the dicts above (`level_proposals_per_image` and
                #      `level_scores_per_image`).
                #
                # NOTE: Your `nms` method may be slow for training - you may
                # use `torchvision.ops.nms` with exact same input arguments,
                # to speed up training. We will grade your `nms` implementation
                # separately; you will NOT lose points if you don't use it here.
                ##############################################################
                # Replace "pass" statement with your code
                boxreg_deltas = level_boxreg_deltas
                ###
                # print('boxreg shape', boxreg_deltas.shape)
                ###
                proposal_boxes = rcnn_apply_deltas_to_anchors(
                    boxreg_deltas, level_anchors
                )
                proposal_boxes[:, :2] = proposal_boxes[:, :2].clamp(min=0)
                proposal_boxes[:, 2] = proposal_boxes[:, 2].clamp(max=image_size[0] - 1)
                proposal_boxes[:, 3] = proposal_boxes[:, 3].clamp(max=image_size[1] - 1)

                obj_logits = level_obj_logits
                sorted, indices = torch.topk(obj_logits, min(self.pre_nms_topk, obj_logits.shape[0]))
                proposal_boxes = proposal_boxes[indices]
                final_idx = torchvision.ops.nms(
                    proposal_boxes, sorted, self.nms_thresh
                )[:self.post_nms_topk]
                proposals_per_fpn_level_per_image[level_name] = proposal_boxes[final_idx]
                scores_per_fpn_level_per_image[level_name] = sorted[final_idx]
                ##############################################################
                #                        END OF YOUR CODE                    #
                ##############################################################

            proposals_all_levels_per_image = self._cat_across_fpn_levels(
                proposals_per_fpn_level_per_image, dim=0
            )
            scores_all_levels_per_image = self._cat_across_fpn_levels(
                scores_per_fpn_level_per_image, dim=0
            )
            # Sort scores from highest to smallest and filter proposals.
            _inds = scores_all_levels_per_image.argsort(descending=True)
            _inds = _inds[: self.post_nms_topk]
            keep_proposals = proposals_all_levels_per_image[_inds]

            proposals_per_image.append(keep_proposals)

        return proposals_per_image


    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)


class FasterRCNN(nn.Module):
    """
    Faster R-CNN detector: this module combines backbone, RPN, ROI predictors.

    Unlike Faster R-CNN, we will use class-agnostic box regression and Focal
    Loss for classification. We opted for this design choice for you to re-use
    a lot of concepts that you already implemented in FCOS - choosing one loss
    over other matters less overall.
    """

    def __init__(
        self,
        backbone: nn.Module,
        rpn: nn.Module,
        stem_channels: List[int],
        num_classes: int,
        batch_size_per_image: int,
        roi_size: Tuple[int, int] = (7, 7),
    ):
        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.batch_size_per_image = batch_size_per_image

        ######################################################################
        # TODO: Create a stem of alternating 3x3 convolution layers and RELU
        # activation modules using `stem_channels` argument, exactly like
        # `FCOSPredictionNetwork` and `RPNPredictionNetwork`. use the same
        # stride, padding, and weight initialization as previous TODOs.
        #
        # HINT: This stem will be applied on RoI-aligned FPN features. You can
        # decide the number of input channels accordingly.
        ######################################################################
        # Fill this list. It is okay to use your implementation from
        # `FCOSPredictionNetwork` for this code block.
        cls_pred = []
        # Replace "pass" statement with your code
        c = [backbone.out_channels, *stem_channels]
        for i in range(len(stem_channels)):
            cls_pred.append(nn.Conv2d(c[i], c[i + 1], 3, padding='same'))
            nn.init.normal_(cls_pred[2 * i].weight, 0, 0.01)
            nn.init.constant_(cls_pred[2 * i].bias, 0)
            cls_pred.append(nn.ReLU())
        ######################################################################
        # TODO: Add an `nn.Flatten` module to `cls_pred`, followed by a linear
        # layer to output C+1 classification logits (C classes + background).
        # Think about the input size of this linear layer based on the output
        # shape from `nn.Flatten` layer.
        ######################################################################
        # Replace "pass" statement with your code
        cls_pred.append(nn.Flatten())
        in_feats = self.roi_size[0] * self.roi_size[1] * c[-1]
        cls_pred.append(nn.Linear(in_feats, num_classes + 1))
        nn.init.normal_(cls_pred[-1].weight, 0.0, 0.01)
        nn.init.constant_(cls_pred[-1].bias, 0.0)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Wrap the layers defined by student into a `nn.Sequential` module,
        # Faster R-CNN also predicts box offsets to "refine" RPN proposals, we
        # exclude it for simplicity and keep RPN proposal boxes as final boxes.
        self.cls_pred = nn.Sequential(*cls_pred)

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        """
        See documentation of `FCOS.forward` for more details.
        """

        feats_per_fpn_level = self.backbone(images)
        output_dict = self.rpn(
            feats_per_fpn_level, self.backbone.fpn_strides, gt_boxes
        )
        
        # List of B (`batch_size`) tensors giving RPN proposals per image.
        proposals_per_image = output_dict["proposals"]

        # Assign the proposals to different FPN levels for extracting features
        # using RoI-align. During training we also mix GT boxes with proposals.
        # NOTE: READ documentation of function to understand what it is doing.
        proposals_per_fpn_level = reassign_proposals_to_fpn_levels(
            proposals_per_image,
            gt_boxes
            # gt_boxes will be None during inference
        )

        # Get batch size from FPN feats:
        num_images = feats_per_fpn_level["p3"].shape[0]

        # Perform RoI-align using FPN features and proposal boxes.
        roi_feats_per_fpn_level = {
            level_name: None for level_name in feats_per_fpn_level.keys()
        }
        # Get RPN proposals from all levels.
        for level_name in feats_per_fpn_level.keys():
            ##################################################################
            # TODO: Call `torchvision.ops.roi_align`. See its documentation to
            # properly format input arguments. Use `aligned=True`
            ##################################################################
            level_feats = feats_per_fpn_level[level_name]
            level_props = proposals_per_fpn_level[level_name]
            level_stride = self.backbone.fpn_strides[level_name]

            # Replace "pass" statement with your code
            roi_feats = torchvision.ops.roi_align(
                level_feats,
                level_props,
                self.roi_size,
                1 / level_stride,
                aligned=True)
            ##################################################################
            #                         END OF YOUR CODE                       #
            ##################################################################

            roi_feats_per_fpn_level[level_name] = roi_feats

        # Combine ROI feats across FPN levels, do the same with proposals.
        # shape: (batch_size * total_proposals, fpn_channels, roi_h, roi_w)
        roi_feats = self._cat_across_fpn_levels(roi_feats_per_fpn_level, dim=0)

        # Obtain classification logits for all ROI features.
        # shape: (batch_size * total_proposals, num_classes)
        pred_cls_logits = self.cls_pred(roi_feats)

        if not self.training:
            # During inference, just go to this method and skip rest of the
            # forward pass. Batch size must be 1!
            # fmt: off
            return self.inference(
                images,
                proposals_per_fpn_level,
                pred_cls_logits,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )
            # fmt: on

        ######################################################################
        # Match the RPN proposals with provided GT boxes and append to
        # `matched_gt_boxes`. Use `rcnn_match_anchors_to_gt` with IoU threshold
        # such that IoU > 0.5 is foreground, otherwise background.
        # There are no neutral proposals in second-stage.
        ######################################################################
        matched_gt_boxes = []
        for _idx in range(len(gt_boxes)):
            # Get proposals per image from this dictionary of list of tensors.
            proposals_per_fpn_level_per_image = {
                level_name: prop[_idx]
                for level_name, prop in proposals_per_fpn_level.items()
            }

            proposals_per_image = self._cat_across_fpn_levels(
                proposals_per_fpn_level_per_image, dim=0
            )
            gt_boxes_per_image = gt_boxes[_idx]
            # Replace "pass" statement with your code
            matched_gt_boxes.append(rcnn_match_anchors_to_gt(
                proposals_per_image, gt_boxes_per_image, (0.5, 0.5)
            ))
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Combine predictions and GT from across all FPN levels.
        matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0)

        ######################################################################
        # TODO: Train the classifier head. Perform these steps in order:
        #   1. Sample a few RPN proposals, like you sampled 50-50% anchor boxes
        #      to train RPN objectness classifier. However this time, sample
        #      such that ~25% RPN proposals are foreground, and the rest are
        #      background. Faster R-CNN performed such weighted sampling to
        #      deal with class imbalance, before Focal Loss was published.
        #
        #   2. Use these indices to get GT class labels from `matched_gt_boxes`
        #      and obtain the corresponding logits predicted by classifier.
        #
        #   3. Compute cross entropy loss - use `F.cross_entropy`, see its API
        #      documentation on PyTorch website. Since background ID = -1, you
        #      may shift class labels by +1 such that background ID = 0 and
        #      other VC classes have IDs (1-20). Make sure to reverse shift
        #      this during inference, so that model predicts VOC IDs (0-19).
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        loss_cls = None
        # Replace "pass" statement with your code
        fg_idx, bg_idx = sample_rpn_training(
            matched_gt_boxes, self.batch_size_per_image * num_images, 0.25
        )

        gt_labels = torch.hstack((
            matched_gt_boxes[fg_idx, -1], matched_gt_boxes[bg_idx, -1]
        )) + 1
        gt_labels = gt_labels.to(dtype=torch.int64)

        fgbg_logits = torch.vstack((
            pred_cls_logits[fg_idx], pred_cls_logits[bg_idx]
        ))
        loss_cls = F.cross_entropy(
            fgbg_logits, gt_labels, reduction="mean"
        )
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        return {
            "loss_rpn_obj": output_dict["loss_rpn_obj"],
            "loss_rpn_box": output_dict["loss_rpn_box"],
            "loss_cls": loss_cls,
        }

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    def inference(
        self,
        images: torch.Tensor,
        proposals: torch.Tensor,
        pred_cls_logits: torch.Tensor,
        test_score_thresh: float,
        test_nms_thresh: float,
    ):
        """
        Run inference on a single input image (batch size = 1). Other input
        arguments are same as those computed in `forward` method. This method
        should not be called from anywhere except from inside `forward`.

        Returns:
            Three tensors:
                - pred_boxes: Tensor of shape `(N, 4)` giving *absolute* XYXY
                  co-ordinates of predicted boxes.

                - pred_classes: Tensor of shape `(N, )` giving predicted class
                  labels for these boxes (one of `num_classes` labels). Make
                  sure there are no background predictions (-1).

                - pred_scores: Tensor of shape `(N, )` giving confidence scores
                  for predictions.
        """

        # The second stage inference in Faster R-CNN is quite straightforward:
        # combine proposals from all FPN levels and perform a *class-specific
        # NMS*. There would have been more steps here if we further refined
        # RPN proposals by predicting box regression deltas.

        # Use `[0]` to remove the batch dimension.
        proposals = {level_name: prop[0] for level_name, prop in proposals.items()}
        pred_boxes = self._cat_across_fpn_levels(proposals, dim=0)

        ######################################################################
        # Faster R-CNN inference, perform the following steps in order:
        #   1. Get the most confident predicted class and score for every box.
        #      Note that the "score" of any class (including background) is its
        #      probability after applying C+1 softmax.
        #
        #   2. Only retain prediction that have a confidence score higher than
        #      provided threshold in arguments.
        #
        # NOTE: `pred_classes` may contain background as ID = 0 (based on how
        # the classifier was supervised in `forward`). Remember to shift the
        # predicted IDs such that model outputs ID (0-19) for 20 VOC classes.
        ######################################################################
        pred_scores, pred_classes = None, None
        # Replace "pass" statement with your code
        scores = F.softmax(pred_cls_logits, dim=1)[:, 1:]
        pred_classes = torch.argmax(scores, dim=1)
        scores = scores[torch.arange(scores.shape[0]), pred_classes]
        conf_idx = scores > test_score_thresh
        pred_scores = scores[conf_idx]
        pred_boxes = pred_boxes[conf_idx]
        pred_classes = pred_classes[conf_idx]
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        # STUDENTS: This line depends on your implementation of NMS.
        keep = class_spec_nms(
            pred_boxes, pred_scores, pred_classes, iou_threshold=test_nms_thresh
        )
        pred_boxes = pred_boxes[keep]
        pred_classes = pred_classes[keep]
        pred_scores = pred_scores[keep]
        return pred_boxes, pred_classes, pred_scores

