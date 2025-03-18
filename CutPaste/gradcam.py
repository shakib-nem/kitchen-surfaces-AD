from typing import Iterable, Optional
import os
import warnings

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import captum.attr
import matplotlib.pyplot as plt

from BurgerDataTest2 import BurgerData as BurgerDataTest
from train2 import CutPaste


class GradCam(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, name_layer: str, mean: list, std: list) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        
        # Extract the desired layer
        names_mid = name_layer.split(".")
        layer = model
        for name in names_mid:
            layer = getattr(layer, name)
        self.layer = layer
        
        # Captum's LayerGradCam
        self.cam = captum.attr.LayerGradCam(self._forward_only_logits, self.layer)
        
        # Mean/std if needed for unnormalization
        self.mean = mean
        self.std  = std

    def _forward_only_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Returns only the logits from the model's output."""
        logits, _ = self.model(x)
        return logits

    def auto_select_indices(self, logits: torch.Tensor) -> torch.Tensor:
        """Pick the top predicted class index for each item."""
        return torch.argmax(logits, dim=1)
    
    def forward(
        self,
        x: torch.Tensor,
        indices: Optional[Iterable[int]] = None,
        with_upsample: bool = False
    ) -> torch.Tensor:
        """
        Returns raw Grad-CAM feature maps (float) of shape [B, 1, H, W] if with_upsample=True.
        We'll color-map them ourselves after reconstructing the full image aggregator.
        """
        if indices is None:
            logits, _ = self.model(x)
            indices = self.auto_select_indices(logits)

        x = x.requires_grad_(True)
        featuremaps = self.cam.attribute(x, target=indices, relu_attributions=True)
        
        if with_upsample:
            featuremaps = F.interpolate(
                featuremaps,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=True
            )
        return featuremaps  # [B, 1, H, W] if upsampled
    
    @staticmethod
    
    def featuremaps_to_heatmaps(x: torch.Tensor) -> np.ndarray:
        """
        Applies per-sample min–max normalization and then uses cv2.applyColorMap(...)
        to generate a heatmap for each feature map.
        
        x: [B, 1, H, W]
        returns: [B, H, W, 3] in BGR
        """
        B, _, H, W = x.shape
        featuremaps = x.squeeze(1).detach().cpu().numpy()
        heatmaps = np.zeros((B, H, W, 3), dtype=np.uint8)

        for i_map, fmap in enumerate(featuremaps):
            # per-sample min–max normalization
            hmap = cv2.normalize(fmap, None, 0, 255, cv2.NORM_MINMAX)
            hmap = cv2.applyColorMap(hmap.astype(np.uint8), cv2.COLORMAP_HOT)
            heatmaps[i_map] = hmap

        return heatmaps
    


if __name__ == "__main__":
    # 1) Dataset configuration
    # Adjust these values as needed.
    w = 2
    if w == 1:
        # white
        mean = [0.5815, 0.5940, 0.5015]    
        std  = [0.2716, 0.2812, 0.2710]
        folder = "/home/shn/data/test/white"
    elif w == 2:
        # white with edges
        mean = [0.6384, 0.6557, 0.5500]
        std  = [0.2846, 0.2897, 0.2772]
        folder = "/home/shn/data/test/white_with_edges"
        #folder = "/home/shn/data/test/edges_2"
    else:
        raise ValueError("No mean and std defined for the selected option.")

    # Initialize dataset and dataloader.
    # Note: The transform is only used for the model input.
    dataset = BurgerDataTest(
        imgSize=244,
        stride=112,
        image_folder=folder,
        transform=transforms.Normalize(mean=mean, std=std)
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 2) Load Model 
    checkpoint_path = '/home/shn/tb_logs/cutAndPaste/version_31/checkpoints/weights.ckpt' #changr versions here to use different models
    model = CutPaste.load_from_checkpoint(checkpoint_path=checkpoint_path).eval().to("cuda")

    # 3) Initialize GradCam
    grad_cam = GradCam(model.model, "encoder.conv1", mean=mean, std=std)

    # We'll store raw Grad-CAM floats in aggregator_sum / aggregator_count
    filename_single = None
    raw_cam_patches = []

    # 4) Collect Grad-CAM patches for full-image reconstruction.
    for patches, labels, file_batch, x_batch, y_batch in dataloader:
        patches = patches.to("cuda")
        # Here we use a fixed target index (e.g., 1). Adjust as needed.
        raw_featuremaps = grad_cam.forward(patches, indices=1, with_upsample=True)
        raw_featuremaps_np = raw_featuremaps.squeeze(1).detach().cpu().numpy()  # shape [B, patchH, patchW]

        for i in range(len(file_batch)):
            fn      = file_batch[i]
            x_coord = x_batch[i].item()
            y_coord = y_batch[i].item()
            fmap_np = raw_featuremaps_np[i]  # float array

            raw_cam_patches.append((fn, x_coord, y_coord, fmap_np))
            if filename_single is None:
                filename_single = fn

    # 5) Load the original image (using PIL converted to RGB).
    img_path = os.path.join(folder, filename_single)
    original_pil = Image.open(img_path).convert("RGB")
    original_np  = np.array(original_pil)
    H, W = original_np.shape[:2]

    # 6) Create aggregators for sum and count (to average overlapping patches)
    aggregator_sum   = np.zeros((H, W), dtype=np.float32)
    aggregator_count = np.zeros((H, W), dtype=np.float32)

    # 7) For each patch, add the Grad-CAM values to the sum and increment the count.
    for (fn, x_coord, y_coord, fmap_np) in raw_cam_patches:
        patch_h, patch_w = fmap_np.shape
        aggregator_sum[y_coord:y_coord+patch_h, x_coord:x_coord+patch_w] += fmap_np
        aggregator_count[y_coord:y_coord+patch_h, x_coord:x_coord+patch_w] += 1

    # 8) Compute the averaged aggregator (avoiding divide-by-zero).
    with np.errstate(divide='ignore', invalid='ignore'):
        aggregator_avg = np.divide(aggregator_sum, np.maximum(aggregator_count, 1))
    aggregator_avg[np.isnan(aggregator_avg)] = 0  # Replace any NaNs with 0

    # 9) Blur the aggregated heatmap to reduce patch boundaries.
    aggregator_blurred = cv2.GaussianBlur(aggregator_avg, ksize=(5, 5), sigmaX=2)

    # 10) Convert the blurred aggregator to a tensor and generate the final heatmap.
    aggregator_ts = torch.from_numpy(aggregator_blurred[None, None, ...])  # => [1, 1, H, W]
    final_heatmap_bhwc = GradCam.featuremaps_to_heatmaps(aggregator_ts)
    final_heatmap = final_heatmap_bhwc[0]  # [H, W, 3] in BGR

    # Convert detection heatmap from BGR to RGB for display.
    detection_rgb = cv2.cvtColor(final_heatmap, cv2.COLOR_BGR2RGB)
    
    

    # 11) Load the ground truth image.
    # we load it from the same folder as the original image.
    # The ground truth image is expected to be named: <filename_without_extension>_groundtruth.png
    gt_img_path = os.path.join(folder, filename_single[:-4] + '_groundtruth.png')
    if os.path.exists(gt_img_path):
        ground_truth_image = Image.open(gt_img_path).convert('L')
        gt_np = np.array(ground_truth_image)
    else:
        print(f"Ground truth not found at {gt_img_path}. Using a blank mask.")
        gt_np = np.zeros((H, W), dtype=np.uint8)

    # 12) Display the 2x2 grid:
    #      Top left: Original image
    #      Top right: Ground truth mask
    #      Bottom left: Detection (Grad-CAM) heatmap
    #      Bottom right: Detection heatmap overlayed on the original image
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Top left: Original image
    axes[0, 0].imshow(original_np)
    axes[0, 0].axis("off")
    axes[0, 0].set_title(f"a. Original Image")

    # Top right: Ground truth mask (displayed with a grayscale colormap)
    axes[0, 1].imshow(gt_np, cmap="gray")
    axes[0, 1].axis("off")
    axes[0, 1].set_title("b. Ground Truth Image")
    
    # Increase contrast by 50%
    alpha = 3  # contrast control (1.0-3.0)
    beta = 20     # brightness control (0-100)
    detection_rgb_contrast = cv2.convertScaleAbs(detection_rgb, alpha=alpha, beta=beta)


    # Create an overlay: blend the detection heatmap with the original image.
    overlay = cv2.addWeighted(original_np, 0.1, detection_rgb_contrast, 1, 0)
    
    # Bottom left: Detection heatmap
    axes[1, 0].imshow(detection_rgb_contrast)
    axes[1, 0].axis("off")
    axes[1, 0].set_title("c. Anomaly Map (Detection Alone)")

    # Bottom right: Detection overlay on the original image
    axes[1, 1].imshow(overlay)
    axes[1, 1].axis("off")
    axes[1, 1].set_title("d. Detection Overlay CutPaste")

    plt.tight_layout()
    plt.show()
