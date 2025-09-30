import cv2
import os
import mediapipe as mp
import argparse

# Function to detect faces and blur them in the image/frame
def process_image(image, face_detection, H, W):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB for MediaPipe
    results = face_detection.process(image_rgb)  # Run face detection

    # If faces are detected, iterate through each detection
    if results.detections is not None:
        for detection in results.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            # Convert relative coordinates to absolute pixel values
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x1 + w * W)
            y2 = int(y1 + h * H)

            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)

            # Blur faces only if the region is valid
            if x2 > x1 and y2 > y1:
                image[y1:y2, x1:x2, :] = cv2.blur(image[y1:y2, x1:x2, :], (50, 50))
    return image

# Parse command-line arguments for model type and file path
args = argparse.ArgumentParser()
args.add_argument("--model", default='video')  # "image" or "video"
args.add_argument("--filePath", default='images-videos/videoss.mp4')  # Path to image or video file
args = args.parse_args()

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    # If image mode is selected, process a single image
    if args.model == "image":
        # Build absolute path to the image file
        image_path = os.path.join(os.path.dirname(__file__), '..', args.filePath)
        image_path = os.path.abspath(image_path)
        image = cv2.imread(image_path)  # Read the image
        if image is None:
            print("Error: Could not read image.")
            exit()
        image = cv2.resize(image, (800, 600))  # Resize for display
        H, W, _ = image.shape

        # Detect and blur faces in the image
        image = process_image(image, face_detection, H, W)

        # Save and display the result
        output_path = os.path.join(os.path.dirname(__file__), '..', 'images-videos', 'face_anonymized.png')
        cv2.imwrite(output_path, image)
        cv2.imshow("Anonymized Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # If video mode is selected, process each frame of the video
    elif args.model == "video":
        print("Processing video...")
        output_folder = os.path.join(os.path.dirname(__file__), '..', 'images-videos')
        os.makedirs(output_folder, exist_ok=True)

        # Build absolute path to the video file
        video_path = os.path.join(os.path.dirname(__file__), '..', args.filePath)
        print("Video path:", video_path)
        cap = cv2.VideoCapture(video_path)  # Open the video file
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        # Get video properties
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 25  # Default FPS if not detected
        print("Frame size:", W, H)

        # Prepare to write the processed video
        output_video = cv2.VideoWriter(
            os.path.join(output_folder, 'face_anonymized.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (W, H)
        )

        # Process each frame of the video
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_image(frame, face_detection, H, W)  # Blur faces in the frame
            output_video.write(frame)  # Write the processed frame to output
            cv2.imshow("Anonymized Video", frame)  # Show the frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()
