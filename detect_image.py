from ultralytics import YOLO
import argparse
import cv2

models = "./models/yolov8n_v2_openvino_model"

def procces_image(input_image: str, confidence_threshold: float = 0.5, iou_threshold: float = 0.5) -> None:
    # Load a pretrained YOLOv8n model
    stub = YOLO(models, task='detect')

    image = cv2.imread(input_image)

    while True:
        # Run inference on 'bus.jpg' with arguments
        results = stub.predict(image, imgsz=640, conf=confidence_threshold, iou=iou_threshold, save=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Object Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process an image using YOLOv8n."
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to the input image."
    )

    args = parser.parse_args()

    procces_image(args.input_image)