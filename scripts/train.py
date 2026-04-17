"""
Training script for YOLO license plate detection
Logs metrics to Weights & Biases (wandb)
"""

import argparse
from pathlib import Path
import os
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO


def setup_wandb(project_name: str, run_name: str, config: dict):
    """
    Setup Weights & Biases logging.
    
    Args:
        project_name: Wandb project name
        run_name: Name of the run
        config: Configuration dictionary to log
    """
    try:
        import wandb
        wandb.init(
            project=project_name,
            name=run_name,
            config=config
        )
        return wandb
    except ImportError:
        print("Wandb not installed. Install with: pip install wandb")
        return None
    except Exception as e:
        print(f"Failed to initialize wandb: {e}")
        return None


def train_model(
    data_yaml: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    model_size: str = 'n',  # n, s, m, l, x
    device: str = 'cuda',
    project: str = 'license-plate-detection',
    name: str = 'yolov8n_plate',
    resume: bool = False,
    use_wandb: bool = True,
    patience: int = 20,
    save_period: int = 10
):
    """
    Train YOLO model for license plate detection.
    
    Args:
        data_yaml: Path to dataset YAML file (from Roboflow)
        epochs: Number of training epochs
        imgsz: Input image size
        batch_size: Batch size
        model_size: YOLO model size (n/s/m/l/x)
        device: Training device ('cuda' or 'cpu')
        project: Project name for saving
        name: Run name
        resume: Resume training from checkpoint
        use_wandb: Whether to use wandb logging
        patience: Early stopping patience
        save_period: Save checkpoint every N epochs
    """
    
    # Check if data file exists
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    
    print("=" * 60)
    print("License Plate Detection Training")
    print("=" * 60)
    print(f"Dataset: {data_yaml}")
    print(f"Model size: YOLOv8{model_size}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print(f"Project: {project}/{name}")
    print("=" * 60)
    
    # Setup wandb
    wandb_logger = None
    if use_wandb:
        config = {
            'epochs': epochs,
            'imgsz': imgsz,
            'batch_size': batch_size,
            'model_size': model_size,
            'device': device,
            'patience': patience
        }
        wandb_logger = setup_wandb(project, name, config)
    
    # Load pretrained model
    model_path = f'yolov8{model_size}.pt'
    print(f"\nLoading pretrained model: {model_path}")
    model = YOLO(model_path)
    
    # Training arguments
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'device': device,
        'project': project,
        'name': name,
        'resume': resume,
        'patience': patience,
        'save': True,
        'save_period': save_period,
        'val': True,
        'plots': True,
        'verbose': True,
        'seed': 42,
        'workers': 8 if device == 'cuda' else 4,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'lr0': 0.01,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate factor
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Class loss gain
        'dfl': 1.5,  # DFL loss gain
        'hsv_h': 0.015,  # Hue augmentation
        'hsv_s': 0.7,   # Saturation augmentation
        'hsv_v': 0.4,   # Value augmentation
        'degrees': 0.0,  # Rotation augmentation
        'translate': 0.1,  # Translation augmentation
        'scale': 0.5,      # Scale augmentation
        'shear': 0.0,      # Shear augmentation
        'perspective': 0.0, # Perspective augmentation
        'flipud': 0.0,     # Flip up-down
        'fliplr': 0.5,     # Flip left-right
        'mosaic': 1.0,     # Mosaic augmentation
        'mixup': 0.0,      # Mixup augmentation
        'copy_paste': 0.0, # Copy-paste augmentation
    }
    
    print("\nStarting training...")
    print("-" * 60)
    
    # Train
    results = model.train(**train_args)
    
    # Log final metrics
    if wandb_logger and results:
        try:
            # Get best metrics
            metrics = {
                'final_map50': results.results_dict.get('metrics/mAP50(B)', 0),
                'final_map': results.results_dict.get('metrics/mAP50-95(B)', 0),
                'final_precision': results.results_dict.get('metrics/precision(B)', 0),
                'final_recall': results.results_dict.get('metrics/recall(B)', 0),
            }
            wandb_logger.log(metrics)
            print("\nFinal metrics logged to wandb:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
        except Exception as e:
            print(f"Failed to log to wandb: {e}")
    
    # Finish wandb session
    if wandb_logger:
        wandb_logger.finish()
    
    # Print results
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    # Find best model
    best_model_path = Path(project) / name / 'weights' / 'best.pt'
    if best_model_path.exists():
        print(f"\nBest model saved to: {best_model_path}")
        
        # Validate best model
        print("\nValidating best model...")
        val_results = model.val(data=data_yaml, device=device)
        print(f"Validation mAP50: {val_results.box.map50:.4f}")
        print(f"Validation mAP50-95: {val_results.box.map:.4f}")
    
    # Export to different formats
    print("\nExporting model...")
    try:
        # Export to ONNX
        onnx_path = model.export(format='onnx', imgsz=imgsz)
        print(f"ONNX model saved to: {onnx_path}")
        
        # Export to TorchScript
        torchscript_path = model.export(format='torchscript', imgsz=imgsz)
        print(f"TorchScript model saved to: {torchscript_path}")
    except Exception as e:
        print(f"Export failed: {e}")
    
    print("\n" + "=" * 60)
    print("Training pipeline finished successfully!")
    print("=" * 60)
    
    return results


def validate_model(
    model_path: str,
    data_yaml: str,
    device: str = 'cpu', 
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
):
    """
    Validate trained model on test dataset.
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to dataset YAML
        device: Device for validation
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
    """
    print("\n" + "=" * 60)
    print("Model Validation")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Dataset: {data_yaml}")
    
    # Load model
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(
        data=data_yaml,
        device=device,
        conf=conf_threshold,
        iou=iou_threshold,
        plots=True,
        save_json=True
    )
    
    # Print metrics
    print("\nValidation Results:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLO license plate detector')
    
    # Required arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset YAML file')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size (default: 640)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--model', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size: n/s/m/l/x (default: n)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience (default: 20)')
    
    # Output options
    parser.add_argument('--project', type=str, default='license-plate-detection',
                       help='Project name (default: license-plate-detection)')
    parser.add_argument('--name', type=str, default=None,
                       help='Run name (default: yolov8{model}_plate)')
    
    # Logging options
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    
    # Validation options
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing model, do not train')
    parser.add_argument('--validate-model', type=str,
                       help='Path to model for validation only')
    
    args = parser.parse_args()
    
    # Set run name
    if args.name is None:
        args.name = f'yolov8{args.model}_plate'
    
    try:
        if args.validate_only:
            # Validation mode
            if not args.validate_model:
                parser.error("--validate-model required for validation mode")
            validate_model(
                model_path=args.validate_model,
                data_yaml=args.data,
                device=args.device
            )
        else:
            # Training mode
            train_model(
                data_yaml=args.data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch_size=args.batch,
                model_size=args.model,
                device=args.device,
                project=args.project,
                name=args.name,
                resume=args.resume,
                use_wandb=not args.no_wandb,
                patience=args.patience
            )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()