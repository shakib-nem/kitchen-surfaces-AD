import argparse
import os
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import auc, roc_curve
from sklearn.neighbors import KernelDensity
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from torchvision import transforms

from model import CutPasteNet
from BurgerDataTrain import BurgerData as BurgerDataTrain
from BurgerDataTest2 import BurgerData as BurgerDataTest
from typing import Any, Dict, Tuple, Union
from sklearn.metrics import accuracy_score , precision_score, recall_score , confusion_matrix
import warnings
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_checkpoints',
                        default='/home/shn/tb_logs/cutAndPaste/version_54/checkpoints', 
                        help='Path to the folder where training checkpoints are saved.')
    parser.add_argument('--train_data',
                        default='/home/shn/data/white/train',
                        help='Path to your training dataset directory.')
    parser.add_argument('--test_data',
                        default='/home/shn/data/white/ground_truth',
                        help='Path to your testing dataset directory.')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--save_exp', default=pathlib.Path(__file__).parent / 'anomaly_exp',
                        help='Directory to save fitted models and ROC curves')
    args = parser.parse_args()
    return args

class AnomalyDetection:
    def __init__(self, weights, batch_size, device='cuda:1'):
        # Initialize the model from saved weights and set device and batch size.
        self.cutpaste_model = self.load_model(weights, device)
        self.device = device
        self.batch_size = batch_size

    @staticmethod
    def load_model(weights, device):
        # Load the CutPasteNet model and its state from the checkpoint.
        model = CutPasteNet(encoder='resnet18', pretrained=False)
        state_dict = torch.load(weights, map_location=device)['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    @staticmethod
    def roc_auc(labels, scores, defect_name=None, save_path=None):
        # Compute the ROC curve and AUC, then save the ROC plot.
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        print("starting ROC")
        plt.figure()
        plt.title(f'ROC curve: {defect_name}')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        
    
        optimal_idx = np.argmax(tpr - fpr)

        optimal_threshold = thresholds[optimal_idx]
        print(f'Optimal Threshold: {optimal_threshold}')

        y_pred_binarized = [1 if score >= optimal_threshold else 0 for score in scores]

        # Compute accuracy, precision, and recall
        accuracy = accuracy_score(labels, y_pred_binarized)
        precision = precision_score(labels, y_pred_binarized,zero_division=1)
        recall = recall_score(labels, y_pred_binarized)
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        
        # Compute and print confusion matrix
        conf_matrix = confusion_matrix(labels, y_pred_binarized)
        print(f'Confusion Matrix:\n{conf_matrix}')
         
        save_images = save_path if save_path else './roc_results'
        os.makedirs(save_images, exist_ok=True)
        image_path = os.path.join(save_images, f'{defect_name}_roc.jpg') if defect_name else os.path.join(save_images, 'roc_curve.jpg')
        plt.savefig(image_path)
        plt.close()
        return roc_auc

    @staticmethod
    def plot_tsne(labels, embeds, defect_name=None, save_path=None, **kwargs):
        """t-SNE visualize

        Args:
            labels (Tensor): labels of test and train
            embeds (Tensor): embeds of test and train
            defect_name ([str], optional): same as <defect_name> in roc_auc. Defaults to None.
            save_path ([str], optional): same as <defect_name> in roc_auc. Defaults to None.
            kwargs (Dict[str, Any]): hyper parameters of t-SNE which will change final result
                n_iter (int): > 250, default = 1000
                learning_rate (float): (10-1000), default = 100
                perplexity (float): (5-50), default = 28
                early_exaggeration (float): change it when not converging, default = 12
                angle (float): (0.2-0.8), default = 0.3
                init (str): "random" or "pca", default = "pca"
        """
        # Configure and run t-SNE for dimensionality reduction.
        tsne = TSNE( 
            n_components=2,
            verbose=1,
            n_iter=kwargs.get("n_iter", 1000),
            learning_rate=kwargs.get("learning_rate", 100),
            perplexity=kwargs.get("perplexity", 30), # in the original file its 28
            early_exaggeration=kwargs.get("early_exaggeration", 12),
            angle=kwargs.get("angle", 0.3),
            init=kwargs.get("init", "pca"),
        )
        embeds, labels = shuffle(embeds, labels)
        tsne_results = tsne.fit_transform(embeds)

        plt.figure()
        plt.title(f't-SNE: {defect_name}')
        for label in np.unique(labels):
            indices = labels == label
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=f'Class {label}')
        plt.legend()
        plt.xticks([])
        plt.yticks([])

        save_images = save_path if save_path else './tsne_results'
        os.makedirs(save_images, exist_ok=True)
        image_path = os.path.join(save_images, f'{defect_name}_tsne.jpg') if defect_name else os.path.join(save_images, 'tsne.jpg')
        plt.savefig(image_path)
        plt.close()

    def create_embeds(self, path_to_images):
        # Compute embeddings for the test dataset.
        embeddings = []
        labels = []
        #white
        mean=[0.5815, 0.5940, 0.5015]
        std=[0.2716, 0.2812, 0.2710]
    
        dataset = BurgerDataTest(
            imgSize=224,
            stride=112,
            image_folder=path_to_images,
            transform=transforms.Normalize(mean=mean, std=std)
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        with torch.no_grad():
            for imgs, lbls in dataloader:
                imgs = imgs.to(self.device)
                _, embeds = self.cutpaste_model(imgs)
                embeddings.append(embeds.cpu())
                labels.append(lbls)
        return torch.cat(embeddings), torch.cat(labels)

    def create_embeds_tsne(self, path_to_images):
        # Calculate embeddings and labels for the training dataset for t-SNE visualization.
        
        print("Calculating t-SNE embeddings and labels...")
        embeddings = []
        labels = []
        #white
        mean=[0.5815, 0.5940, 0.5015]
        std=[0.2716, 0.2812, 0.2710]
        
        dataset = BurgerDataTrain(
            imgSize=224,
            stride=112,
            image_folder=path_to_images,
            transform=transforms.Normalize(mean = mean, std=std)
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                print(f"Batch {batch_idx}:")

                batch_images, batchbad = batch  # Unpack images and labels
                images = torch.vstack([batch_images, batchbad])  # (batchsize * 2, 3, imgsize, imgsize)
                
                labels_batch = torch.zeros(images.shape[0])
                labels_batch[images.shape[0] // 2:] = 1

                images = images.to(self.device)
                _, embeds = self.cutpaste_model(images)
                embeddings.append(embeds.cpu())
                labels.append(labels_batch.cpu())

        embeddings = torch.cat(embeddings)
        labels = torch.cat(labels)

        return embeddings, labels

    @staticmethod
    def GDE_fit(train_embeds, save_path=None):
        # Fit a Kernel Density Estimator (GDE) on the training embeddings.
        GDE = KernelDensity().fit(train_embeds)
        if save_path:
            filename = os.path.join(save_path, 'GDE.sav')
            pickle.dump(GDE, open(filename, 'wb'))
        return GDE

    @staticmethod
    def GDE_scores(embeds, GDE):
        # Compute and normalize anomaly scores using the fitted GDE.
        scores = GDE.score_samples(embeds)
        norm = np.linalg.norm(-scores)
        return -scores / norm

    def anomaly_detection(self, train_data_path, test_data_path, save_path=None):
        """
        Main function to perform anomaly detection:
        - Compute embeddings for training and testing data.
        - Fit the GDE model on training embeddings.
        - Compute anomaly scores for test embeddings.
        - Evaluate and visualize results using ROC and t-SNE.
        """
        train_images = train_data_path
        test_images = test_data_path
        defect_name = os.path.basename(test_data_path)
        print(f"Using training data path: {train_images}")
        print(f"Using testing data path: {test_images}")

        
        # Check if directories exist
        if not os.path.isdir(train_images):
            raise FileNotFoundError(f"Training data directory does not exist: {train_images}")
        if not os.path.isdir(test_images):
            raise FileNotFoundError(f"Testing data directory does not exist: {test_images}")
        
        # Create embeddings for training data
        train_embeds, train_labels = self.create_embeds_tsne(train_images)
        print("done with train embeddings")
        
        # Fit the GDE model
        GDE_model = self.GDE_fit(train_embeds, save_path)
        print("done with GDE model")
        
        # Create embeddings for test data
        test_embeds, test_labels = self.create_embeds(test_images)
        print("done with test embeddings and labels")
        
        # Compute anomaly scores
        scores = self.GDE_scores(test_embeds, GDE_model)
        print("done with GDE scores")
        
        # Compute AUC and plot ROC curve
        auc_score = self.roc_auc(test_labels, scores, defect_name, save_path)
        
        # Plot t-SNE visualization
        self.plot_tsne(torch.cat([train_labels, test_labels]), torch.cat([train_embeds, test_embeds]), defect_name, save_path)
        return auc_score

if __name__ == '__main__':
    #warnings.filterwarnings("ignore") # supress warnings on terminal
    start_time = datetime.now()
    args = get_args()
    print(f"Using path_to_checkpoints: {args.path_to_checkpoints}")
    print(f"Using training data path: {args.train_data}")
    print(f"Using testing data path: {args.test_data}")

    # Get the list of checkpoint files
    all_checkpoints = sorted(pathlib.Path(args.path_to_checkpoints).glob("**/*.ckpt"))
    checkpoint = str(all_checkpoints[0]) if all_checkpoints else None
    if not checkpoint:
        raise FileNotFoundError("No checkpoint files found in the specified path.")

    # Initialize the anomaly detection with the checkpoint
    anomaly_detector = AnomalyDetection(checkpoint, int(args.batch_size))

    # Define paths to your train and test data
    train_data_path = args.train_data
    test_data_path = args.test_data
    save_path = args.save_exp
    os.makedirs(save_path, exist_ok=True)

    # Run anomaly detection
    auc_score = anomaly_detector.anomaly_detection(train_data_path, test_data_path, save_path)
    print(f'AUC = {auc_score}, ROC curve and t-SNE plot saved in: {save_path}')
    with open(os.path.join(save_path, 'AUC_results.txt'), 'a') as f:
        f.write(f'AUC Score: {auc_score}\n')
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    # Format the time difference
    print(f"Execution started at {start_time} and ended at {end_time}")
    print(f"Total duration: {elapsed_time}")
