import cv2
import numpy as np
from ultralytics import YOLO
import os

# Function to simulate a traffic light on the frame at a responsive position
def simulate_traffic_light(frame, light_state, position):
    """
    Simulates a traffic light at the given position on the frame.
    Args:
        frame: The video frame to draw the traffic light on.
        light_state: Current state of the light (0=Red, 1=Yellow, 2=Green).
        position: (x, y) coordinates of the top-left corner of the traffic light.
    """
    traffic_light_colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0)]  # Red, Yellow, Green
    x, y = position
    # Draw the traffic light housing
    cv2.rectangle(frame, (x, y), (x + 50, y + 150), (50, 50, 50), -1)
    # Draw the lights
    for i, color in enumerate(traffic_light_colors):
        cv2.circle(frame, (x + 25, y + 25 + i * 50), 15, color if i == light_state else (50, 50, 50), -1)

# Function to determine signal color based on vehicle count
def determine_signal(total_vehicles):
    """
    Determine the signal color and corresponding traffic light state based on vehicle count.
    Args:
        total_vehicles: Total number of detected vehicles.
    Returns:
        Tuple of signal color name, BGR color, and light state index.
    """
    if total_vehicles < 10:
        return "Green", (0, 255, 0), 2  # Green light
    elif total_vehicles < 20:
        return "Yellow", (0, 255, 255), 1  # Yellow light
    else:
        return "Red", (0, 0, 255), 0  # Red light

# Function to load videos from a folder
def load_videos_from_folder(video_input):
    """Load all video files from the provided folder."""
    videos = []
    if os.path.isfile(video_input) and video_input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        videos.append({"path": video_input, "road_name": os.path.basename(video_input)})
    elif os.path.isdir(video_input):
        for filename in os.listdir(video_input):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                videos.append({"path": os.path.join(video_input, filename), "road_name": filename})
    else:
        print(f"Invalid path or no video files found: {video_input}")
    return videos

# Function to process individual frames
def process_frame(frame, model, road_name, directions):
    """
    Process a video frame: Detect objects, classify directions, and overlay information.
    Args:
        frame: The input video frame.
        model: The YOLO model for object detection.
        road_name: Name of the road the video corresponds to.
        directions: Dictionary to count vehicle directions.
    Returns:
        Processed frame with detection overlays and vehicle count per direction.
    """
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()
    vehicle_count = {"car": 0, "truck": 0, "motorcycle": 0, "bus": 0}
    
    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        class_id = int(class_id)
        label = model.names[class_id]
        
        if label in vehicle_count:
            vehicle_count[label] += 1
            center_x = (x1 + x2) / 2
            if center_x < frame.shape[1] / 2:
                directions['left'] += 1
            else:
                directions['right'] += 1

            color = (0, 255, 0) if conf > 0.5 else (0, 0, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Add road name and vehicle information to the frame
    cv2.putText(frame, f"Road: {road_name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset = 50
    for direction, count in directions.items():
        cv2.putText(frame, f"{direction.capitalize()}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 25

    total_vehicles = sum(vehicle_count.values())
    cv2.putText(frame, f"Total Vehicles: {total_vehicles}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 25

    # Determine signal and overlay the traffic light
    signal_color, signal_rgb, light_state = determine_signal(total_vehicles)
    cv2.putText(frame, f"Signal: {signal_color}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, signal_rgb, 2)
    
    # Adjust position of traffic light based on frame size
    traffic_light_position = (frame.shape[1] - 100, 20)  # Right top corner
    simulate_traffic_light(frame, light_state, position=traffic_light_position)

    return frame, vehicle_count

# Function to process videos
def process_atcc_videos(video_input, model):
    """Process videos from the input files and display results."""
    input_files = load_videos_from_folder(video_input)
    if not input_files:
        return
    caps = []
    road_names = []

    for file_info in input_files:
        cap = cv2.VideoCapture(file_info["path"])
        if not cap.isOpened():
            print(f"Error: Could not open video stream for {file_info['path']}")
            continue
        caps.append(cap)
        road_names.append(file_info["road_name"])

    # Define the target size for all frames (ensure consistent dimensions for stacking)
    target_width, target_height = 640, 480

    while True:
        processed_frames = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret: # if not true
                print(f"End of video stream for {road_names[i]}.")
                caps[i].release()
                continue
            directions = {"left": 0, "right": 0}
            frame_resized = cv2.resize(frame, (target_width, target_height))
            processed_frame, _ = process_frame(frame_resized, model, road_names[i], directions)
            processed_frames.append(processed_frame)

        if len(processed_frames) == 0:
            break

        # Horizontal stacking (2 per row)
        rows = []
        for i in range(0, len(processed_frames), 2):#[v1,v2,v3,v4,v5] v1->v3->v5 ,[v1]
            if i + 1 < len(processed_frames): # 1 < 5
                row = np.hstack(processed_frames[i:i + 2])#0:2 -> 0,1-    v1,v2     v3,v4   v5
            else:
                row = processed_frames[i]
            rows.append(row)

        # Resize all rows to same width and height
        max_width = max(row.shape[1] for row in rows)
        max_height = max(row.shape[0] for row in rows)
        rows_resized = [cv2.resize(row, (max_width, max_height)) for row in rows]

        # Vertical stacking
        grid_frame = np.vstack(rows_resized)
        cv2.imshow("Traffic Management System", grid_frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

#main execution
if __name__ == "__main__":
    model = YOLO("atcc.pt")
    # Single video:
    video_input = r"C:\Users\Admin\ANPR-and-ATCC-for-Smart-Traffic-Management\sample detection videos\atcc sample.mp4"


    #Folder of videos:
    #video_input =r"C:\Users\cheta\Desktop\4"
    process_atcc_videos(video_input, model)
