from ultralytics import YOLO
import os

def analyze_training(logs):
    """Analyze training logs to detect overfitting/underfitting/balanced."""
    train_losses = logs['train/box_loss']
    val_maps = logs['metrics/mAP50']

    if len(train_losses) < 2 or len(val_maps) < 2:
        return "Not enough data yet"

    if train_losses[-1] < train_losses[0] and val_maps[-1] < val_maps[-2]:
        return "⚠ Overfitting: Training loss ↓ but validation mAP dropped"
    elif train_losses[-1] > 1.5 and val_maps[-1] < 0.3:
        return "⚠ Underfitting: Both training loss & validation mAP are poor"
    else:
        return "✅ Balanced: Model is learning & generalizing"

def train_model():
    # Load a pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Train the model
    results = model.train(
        data="doclaynet.yaml",
        imgsz=640,
        batch=8,
        epochs=20,
        patience=4,
        device=0,  # Use GPU if available
        workers=2,
        amp=True,
        pretrained=True,
        project="runs",
        name="doclaynet_v8n_fast",
        freeze=10,
        verbose=True
    )

    print("Model training complete.")
    print(f"Results saved to: {results.save_dir}")

    # Load metrics from results object for analysis
    if hasattr(results, 'results_dicts'):
        for epoch_idx, log in enumerate(results.results_dicts):
            print(f"Epoch {epoch_idx+1}: "
                  f"Train Box Loss={log['train/box_loss']:.4f}, "
                  f"Train Cls Loss={log['train/cls_loss']:.4f}, "
                  f"Train DFL Loss={log['train/dfl_loss']:.4f}, "
                  f"Val mAP50={log['metrics/mAP50']:.4f}")
            print(analyze_training({
                'train/box_loss': [r['train/box_loss'] for r in results.results_dicts[:epoch_idx+1]],
                'metrics/mAP50': [r['metrics/mAP50'] for r in results.results_dicts[:epoch_idx+1]]
            }))
            print("-" * 50)

    # Evaluate the model
    print("Evaluating the model...")
    model.val(data="doclaynet.yaml", imgsz=640)
    print("Model evaluation complete.")

if __name__ == "__main__":
    train_model()
