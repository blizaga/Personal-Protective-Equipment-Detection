import cv2
from ultralytics import YOLO
import time
import cv2
import time
import argparse

models = "./models/yolov8n_v2_openvino_model"
def process_video(
    input_video: str,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.5,
) -> None:
    
    # Load the YOLOv8 model
    model = YOLO(models, task='detect')

    video_path = input_video
    cap = cv2.VideoCapture(video_path)


    # Initialize variables for FPS calculation
    start_time = time.time()
    frame_count = 0

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success==True:
            frame_count += 1
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            lsub_face = cv2.resize(frame, (640, 480))
            results = model.track(lsub_face, conf=confidence_threshold, iou=iou_threshold)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Calculate FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time

            # Draw FPS on the frame
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object, output video writer, and close the display window
    cap.release()
    #output.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video Processing with YOLOv8"
    )
    parser.add_argument(
        "--source",
        dest="input_video",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        dest="confidence_threshold",
        default=0.35,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold",
        dest="iou_threshold", 
        default=0.5, 
        help="IOU threshold for the model",
        type=float
    )

    args = parser.parse_args()

    process_video(
        input_video=args.input_video,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold
    )