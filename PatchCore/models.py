from typing import Tuple
from tqdm import tqdm

import torch
from torch import tensor
from torch.utils.data import DataLoader
import timm

import numpy as np
from sklearn.metrics import roc_auc_score

from .utils import GaussianBlur, get_coreset_idx_randomp, get_tqdm_params


class KNNExtractor(torch.nn.Module):
	def __init__(
		self,
		backbone_name : str = "resnet50",
		out_indices : Tuple = None,
		pool_last : bool = False,
	):
		super().__init__()

		self.feature_extractor = timm.create_model(
			backbone_name,
			out_indices=out_indices,
			features_only=True,
			pretrained=True,
		)
		for param in self.feature_extractor.parameters():
			param.requires_grad = False
		self.feature_extractor.eval()
		
		self.pool = torch.nn.AdaptiveAvgPool2d(1) if pool_last else None
		self.backbone_name = backbone_name # for results metadata
		self.out_indices = out_indices

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.feature_extractor = self.feature_extractor.to(self.device)
			
	def __call__(self, x: tensor):
		with torch.no_grad():
			feature_maps = self.feature_extractor(x.to(self.device))
		feature_maps = [fmap.to("cpu") for fmap in feature_maps]
		if self.pool:
			# spit into fmaps and z
			return feature_maps[:-1], self.pool(feature_maps[-1])
		else:
			return feature_maps

	def fit(self, _: DataLoader):
		raise NotImplementedError

	def predict(self, _: tensor):
		raise NotImplementedError

	def evaluate(self, test_dl: DataLoader) -> Tuple[float, float]:
		"""Calls predict step for each test sample."""
		image_preds = []
		image_labels = []
		pixel_preds = []
		pixel_labels = []

		for sample, mask, label in tqdm(test_dl, **get_tqdm_params()):
			z_score, fmap = self.predict(sample)
			
			image_preds.append(z_score.numpy())
			image_labels.append(label)
			
			pixel_preds.extend(fmap.flatten().numpy())
			pixel_labels.extend(mask.flatten().numpy())
			
		image_labels = np.stack(image_labels)
		image_preds = np.stack(image_preds)

		image_rocauc = roc_auc_score(image_labels, image_preds)
		pixel_rocauc = roc_auc_score(pixel_labels, pixel_preds)

		return image_rocauc, pixel_rocauc

	def get_parameters(self, extra_params : dict = None) -> dict:
		return {
			"backbone_name": self.backbone_name,
			"out_indices": self.out_indices,
			**extra_params,
		}

class PatchCore(KNNExtractor):
	def __init__(
		self,
		f_coreset: float = 0.01, # what percentage of the memory bank to use for coreset
		backbone_name : str = "resnet18",
		coreset_eps: float = 0.90, # sparse projection parameter 
	):
		super().__init__(
			backbone_name=backbone_name,
			out_indices=(2,3),
		)
		self.f_coreset = f_coreset
		self.coreset_eps = coreset_eps
		self.image_size = 224
		self.average = torch.nn.AvgPool2d(3, stride=1) #3 is neighborhood size and 1 is stride (no skipping), increasing the stride will yield worse results according to the paper
		self.blur = GaussianBlur(4)
		self.n_reweight = 3

		self.patch_lib = []
		self.resize = None

	def fit(self, train_dl):
		for sample, _ in tqdm(train_dl, **get_tqdm_params()):
			feature_maps = self(sample)

			if self.resize is None:
				largest_fmap_size = feature_maps[0].shape[-2:] #this return the H and W at level j+1 (3)
				self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)
			resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
			patch = torch.cat(resized_maps, 1)
			patch = patch.reshape(patch.shape[1], -1).T
			self.patch_lib.append(patch)

		self.patch_lib = torch.cat(self.patch_lib, 0)
		
  		# coreset subsampling
		if self.f_coreset < 1:
			self.coreset_idx = get_coreset_idx_randomp(
				self.patch_lib,
				n=int(self.f_coreset * self.patch_lib.shape[0]),
				eps=self.coreset_eps,
    			#force_cpu=True,#added by me
			)
			self.patch_lib = self.patch_lib[self.coreset_idx]

	def predict(self, sample):
		feature_maps = self(sample) #applies the feature extractor (ResNet layers at levels j and j+1) to the input image
     	#added by me
		self.resize = torch.nn.AdaptiveAvgPool2d(feature_maps[0].shape[-2:])
  		
		resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
		patch = torch.cat(resized_maps, 1) #combines features from both levels j and j+1
		patch = patch.reshape(patch.shape[1], -1).T #creates a 2D tensor where: rows = pixels, columns = features

		dist = torch.cdist(patch, self.patch_lib) #calculates the pairwise distances between the input patch and the memory bank
		min_val, min_idx = torch.min(dist, dim=1) #finds the closest neighbour in the memory bank
		s_idx = torch.argmax(min_val) #finds the most anomalous patch
		s_star = torch.max(min_val) # calculates the anomaly score

		# reweighting
		m_test = patch[s_idx].unsqueeze(0) # anomalous patch
		m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0) # its closest neighbour
		w_dist = torch.cdist(m_star, self.patch_lib) # find knn to m_star pt.1
		_, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False) # pt.2
		
		m_star_knn = torch.linalg.norm(m_test-self.patch_lib[nn_idx[0,1:]], dim=1)
		# Softmax normalization trick as in transformers.
		# As the patch vectors grow larger, their norm might differ a lot.
		# exp(norm) can give infinities.
		D = torch.sqrt(torch.tensor(patch.shape[1]))
		w = 1-(torch.exp(s_star/D)/(torch.sum(torch.exp(m_star_knn/D))))
		s = w*s_star 

		# segmentation map
		s_map = min_val.view(1,1,*feature_maps[0].shape[-2:]) #reshape the anomaly score to the shape of the input image
		s_map = torch.nn.functional.interpolate( # resize the anomaly score to the input image size
			s_map, size=(self.image_size,self.image_size), mode='bilinear'
		)
		s_map = self.blur(s_map) # apply Gaussian blur to the anomaly score

		return s, s_map #return the anomaly score and the segmentation map


	def get_parameters(self):
		return super().get_parameters({
			"f_coreset": self.f_coreset,
			"n_reweight": self.n_reweight,
		})
