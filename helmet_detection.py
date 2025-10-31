import os
from ultralytics import YOLO
import cv2

def main_fun(video_path):
    # Mapping class IDs to labels 
    id2class_map = {
        0: 'with helmet',
        1: 'without helmet',
        2: 'rider',
        3:'car'
    }

    # Load YOLO model
    model_path = r"C:\Users\Admin\ANPR-and-ATCC-for-Smart-Traffic-Management\helmet.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    model = YOLO(model_path)

    # Check if video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"Processing video: {os.path.basename(video_path)}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Error opening video stream or file: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        # Run YOLO prediction on the current frame
        results = model.predict(source=frame, imgsz=640, conf=0.5, verbose=False)

        # Annotate the frame with detection results
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("Helmet Detection", annotated_frame)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Interrupted by user.")
            break

    
    cap.release()
    cv2.destroyAllWindows()
    print("Finished processing video.")

# calling the main function
if __name__ == "__main__":
   video_path = r"C:\Users\Admin\ANPR-and-ATCC-for-Smart-Traffic-Management\sample detection videos\helmet detection.mp4"
   main_fun(video_path)

