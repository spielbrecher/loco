from ultralytics import YOLO
from ultralytics import settings


def train():
    # Create a new YOLO model from scratch
    # model = YOLO('yolov8n.yaml')
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')
    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data='config.yaml', epochs=1000)
    # Evaluate the model's performance on the validation set
    results = model.val()

    # Perform object detection on an image using the model
    results = model('test.jpg')
    print(results)
    # Export the model to ONNX format
    success = model.export(format='onnx')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train()


