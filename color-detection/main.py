import cv2
from PIL import Image

from util import get_limits

# Define the target color in BGR format (yellow)
yellow = [0, 255, 255]  # BGR

# Start capturing video from the default camera (webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame from BGR to HSV color space
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the lower and upper HSV limits for the target color
    lowerLimit, upperLimit = get_limits(yellow)

    # Create a mask that isolates the target color in the frame
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    # Convert the mask (NumPy array) to a PIL Image object
    mask_im = Image.fromarray(mask)

    # Get the bounding box of the non-zero regions in the mask
    bbox = mask_im.getbbox()

    # If a bounding box is found (i.e., the target color is detected)
    if bbox is not None:
        x1, y1, x2, y2 = bbox  # Extract the coordinates of the bounding box
        # Draw a blue rectangle around the detected region on the original frame
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display the mask in a window
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()