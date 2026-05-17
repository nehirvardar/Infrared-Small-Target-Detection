"""COCO JSON Format utilities for saving results and creating visualizations"""

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import os


class COCOFormatter:
    """Convert metric results to COCO JSON format"""
    
    def __init__(self, dataset_name, model_name, epoch):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.epoch = epoch
        self.timestamp = datetime.now().isoformat()
        
    def create_metrics_json(self, metrics_dict):
        """Create COCO-style metrics JSON"""
        coco_json = {
            "info": {
                "description": f"Metrics for {self.model_name} on {self.dataset_name}",
                "dataset": self.dataset_name,
                "model": self.model_name,
                "epoch": self.epoch,
                "date": self.timestamp
            },
            "metrics": metrics_dict
        }
        return coco_json
    
    def create_results_json(self, recall, precision, bbox_recall=None, bbox_precision=None, 
                           mean_iou=None, bbox_mAP=None, test_loss=None, train_loss=None):
        """Create comprehensive results JSON"""
        
        # Calculate metrics
        pixel_mAP = float(np.trapezoid(precision, recall)) if len(recall) > 0 else 0.0
        
        results = {
            "info": {
                "description": f"Evaluation results for {self.model_name}",
                "dataset": self.dataset_name,
                "model": self.model_name,
                "epoch": self.epoch,
                "timestamp": self.timestamp
            },
            "metrics": {
                "pixel_based": {
                    "recall": recall.tolist() if isinstance(recall, np.ndarray) else recall,
                    "precision": precision.tolist() if isinstance(precision, np.ndarray) else precision,
                    "mAP": float(pixel_mAP),
                    "mean_IoU": float(mean_iou) if mean_iou is not None else None,
                    "test_loss": float(test_loss) if test_loss is not None else None,
                    "train_loss": float(train_loss) if train_loss is not None else None,
                },
                "bounding_box": {
                    "recall": bbox_recall.tolist() if isinstance(bbox_recall, np.ndarray) else bbox_recall,
                    "precision": bbox_precision.tolist() if isinstance(bbox_precision, np.ndarray) else bbox_precision,
                    "mAP": float(bbox_mAP) if bbox_mAP is not None else None,
                    "iou_threshold": 0.1
                } if bbox_recall is not None else None
            }
        }
        return results


class MetricsVisualizer:
    """Create precision-recall curves and other visualizations"""
    
    @staticmethod
    def plot_pr_curves(recall, precision, bbox_recall=None, bbox_precision=None, 
                       save_dir=None, filename='pr_curves.png'):
        """Plot precision-recall curves"""
        fig, axes = plt.subplots(1, 2 if bbox_recall is not None else 1, figsize=(14, 5))
        
        # Pixel-based metrics
        ax = axes[0] if bbox_recall is not None else axes
        sorted_indices = np.argsort(recall)
        sorted_recall = recall[sorted_indices]
        sorted_precision = precision[sorted_indices]
        mAP = np.trapezoid(sorted_precision, sorted_recall)
        
        ax.plot(sorted_recall, sorted_precision, marker='o', linewidth=2, label=f'Pixel-based (mAP={mAP:.4f})')
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve (Pixel-based)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Bounding box metrics
        if bbox_recall is not None and bbox_precision is not None:
            ax = axes[1]
            bbox_sorted_indices = np.argsort(bbox_recall)
            bbox_sorted_recall = bbox_recall[bbox_sorted_indices]
            bbox_sorted_precision = bbox_precision[bbox_sorted_indices]
            bbox_mAP = np.trapezoid(bbox_sorted_precision, bbox_sorted_recall)
            
            ax.plot(bbox_sorted_recall, bbox_sorted_precision, marker='s', linewidth=2, 
                   color='orange', label=f'BBox-based (mAP={bbox_mAP:.4f})')
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_title('Precision-Recall Curve (Bounding Box, IoU=0.1)', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, filename)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved PR curves to {save_path}")
        
        plt.close()
        return fig
    
    @staticmethod
    def plot_metrics_comparison(recalls_history, precisions_history, save_dir=None, filename='metrics_history.png'):
        """Plot metrics over epochs"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 9))
        
        epochs = range(len(recalls_history))
        
        # Recall plot
        ax = axes[0]
        for i, (threshold_recalls) in enumerate(recalls_history):
            if i == 0 or i == len(recalls_history) - 1:
                ax.plot(epochs, threshold_recalls, marker='o', label=f'Threshold {i/10:.1f}', alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Recall', fontsize=12)
        ax.set_title('Recall over Epochs (Selected Thresholds)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Precision plot
        ax = axes[1]
        for i, (threshold_precisions) in enumerate(precisions_history):
            if i == 0 or i == len(precisions_history) - 1:
                ax.plot(epochs, threshold_precisions, marker='s', label=f'Threshold {i/10:.1f}', alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision over Epochs (Selected Thresholds)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, filename)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved metrics history to {save_path}")
        
        plt.close()
        return fig
    
    @staticmethod
    def plot_single_threshold(recall_values, precision_values, save_dir=None, filename='threshold_metrics.png'):
        """Plot recall vs precision at different thresholds"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        thresholds = np.linspace(0, 1, len(recall_values))
        ax.plot(thresholds, recall_values, marker='o', linewidth=2, label='Recall', color='blue')
        ax.plot(thresholds, precision_values, marker='s', linewidth=2, label='Precision', color='red')
        
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Recall and Precision vs Threshold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, filename)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved threshold metrics to {save_path}")
        
        plt.close()
        return fig


def save_results_coco_json(dataset_dir, st_model, epoch, mean_iou, recall, precision, 
                          bbox_recall=None, bbox_precision=None, bbox_mAP=None, 
                          test_loss=None, train_loss=None, create_plots=True, use_bbox_as_primary=False):
    """
    Save results in COCO JSON format and create visualizations
    
    Args:
        dataset_dir: Path to dataset directory
        st_model: Model name
        epoch: Epoch number
        mean_iou: Mean IoU value
        recall: Recall array (BBox if use_bbox_as_primary=True)
        precision: Precision array (BBox if use_bbox_as_primary=True)
        bbox_recall: Bounding box recall array (not used if use_bbox_as_primary=True)
        bbox_precision: Bounding box precision array (not used if use_bbox_as_primary=True)
        bbox_mAP: Bounding box mAP
        test_loss: Test loss value
        train_loss: Training loss value
        create_plots: Whether to create visualization plots
        use_bbox_as_primary: If True, treat recall/precision as BBox metrics
    """
    
    value_result_dir = os.path.join(dataset_dir, 'value_result')
    os.makedirs(value_result_dir, exist_ok=True)
    
    # Create formatter
    formatter = COCOFormatter(
        dataset_name=dataset_dir.split('/')[-1],
        model_name=st_model,
        epoch=epoch
    )
    
    # Create results JSON - BBox as primary if specified
    if use_bbox_as_primary:
        results_json = formatter.create_results_json(
            recall, precision, None, None,
            mean_iou, bbox_mAP, test_loss, train_loss
        )
    else:
        results_json = formatter.create_results_json(
            recall, precision, bbox_recall, bbox_precision,
            mean_iou, bbox_mAP, test_loss, train_loss
        )
    
    # Save JSON
    json_path = os.path.join(value_result_dir, f'{st_model}_results_epoch{epoch:04d}_coco.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved COCO JSON results to {json_path}")
    
    # Create visualizations
    if create_plots:
        visualizer = MetricsVisualizer()
        
        # PR curves - use bbox if primary, otherwise pixel-based
        if use_bbox_as_primary:
            visualizer.plot_pr_curves(recall, precision, None, None,
                                     value_result_dir, f'{st_model}_pr_curves_epoch{epoch:04d}.png')
        else:
            visualizer.plot_pr_curves(recall, precision, bbox_recall, bbox_precision,
                                     value_result_dir, f'{st_model}_pr_curves_epoch{epoch:04d}.png')
        
        # Threshold metrics
        threshold_path = os.path.join(value_result_dir, f'{st_model}_threshold_epoch{epoch:04d}.png')
        visualizer.plot_single_threshold(recall, precision,
                                        value_result_dir, f'{st_model}_threshold_epoch{epoch:04d}.png')
    
    return results_json
