from ultralytics import YOLO

def main():
    # Load the YOLOv11 classification nano model (it will auto-download 'yolo11n-cls.pt')
    model = YOLO("yolo11n-cls.pt")

    # Train the model using your structured image folders
    results = model.train(
        data="./drone_images",  # Points straight to your folder structure
        epochs=15,              # Number of training loops
        imgsz=224,              # Standard size for image classification
        batch=16,               # Batch size optimized for your hardware
        name="drone_classifier" # Output folder name inside runs/classify/
    )

    print("Training complete!")

if __name__ == '__main__':
    main()