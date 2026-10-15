# Google Colab: Hyperparameter Tuning / Evolution for YOLO11s
# Upload this script or copy-paste it into a Google Colab notebook cell.
# Ensure you are using the T4 GPU runtime.

from ultralytics import YOLO

def main():
    print("Starting YOLO11s Hyperparameter Evolution on Google Colab...")
    
    # Initialize YOLO11s model
    model = YOLO('yolo11s.pt')
    
    # Run Hyperparameter Tuning
    # 'tune' automatically uses Ray Tune or YOLO's built-in genetic algorithm
    # to evolve hyperparameters like learning rate, momentum, augmentation scales, etc.
    # It runs for 30 epochs per iteration, for 30 iterations to find the absolute best pattern.
    results = model.tune(
        data='/content/drive/MyDrive/unified_yolov8_dataset/data.yaml',  # Path to your dataset
        epochs=30,
        iterations=30,
        optimizer='AdamW',
        plots=True,
        save=False,
        val=False,
        device=0
    )
    
    print("Evolution complete! The best hyperparameters are saved in runs/detect/tune/best_hyperparameters.yaml")
    
if __name__ == '__main__':
    # Required setup for Colab before running the Python code:
    # !pip install ultralytics ray[tune]
    # from google.colab import drive
    # drive.mount('/content/drive')
    
    main()
