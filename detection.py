import cv2
import numpy as np

def detect_by_color(frame, lower_color, upper_color):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask based on the color range
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (assuming it's the washer)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw a bounding box around the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Return the bounding box as the region of interest (ROI)
        return (x, y, x + w, y + h), frame

    return None, frame  # No washer detected, return None


def detect_circle(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve circle detection
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Use HoughCircles to detect circles in the frame
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=100)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")  # Convert to integer
        for (x, y, r) in circles:
            # Draw the bounding box (circle) around the detected washer
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)  # Draw circle
            cv2.rectangle(frame, (x - r, y - r), (x + r, y + r), (0, 128, 255), 2)  # Draw bounding box
            
            # Return the bounding box as the region of interest (ROI)
            return (x - r, y - r, x + r, y + r), frame
    
    return None, frame  # No washer detected, return None


def detect_object(video=0):

    '''
    Call this function from the main trakcer module. 
    It allows the user to select the initial bounding box live.
    '''

    # Open the video capture (0 for default webcam, or provide video file path)
    cap = cv2.VideoCapture(video)

    # cap = cv2.VideoCapture("IMG_2585.MOV")

    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    # Define the lower and upper bounds for the washer's color in HSV space
    # These values need to be fine-tuned based on the actual washer color
    lower_color = np.array([100, 150, 0])  # Example values (adjust these)
    upper_color = np.array([140, 255, 255])  # Example values (adjust these)

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to capture frame from video.")
            break

        roi, frame_with_bbox = detect_by_color(frame, lower_color, upper_color)
        if roi is not None:
            cv2.putText(frame_with_bbox, f"Washer detected at: {roi}", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,0,255),2)

        # Display the frame with the bounding box (if detected)
        frame_height, frame_width = frame_with_bbox.shape[:2]
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detection", min(frame_width, 800), min(frame_height, 600))  # Limit window size
        cv2.imshow('Detection', frame_with_bbox)

        # Exit the loop when 'Enter' is pressed
        if cv2.waitKey(1) & 0xFF == 13:
            # Release the video capture object and close all OpenCV windows
            cap.release()
            cv2.destroyAllWindows()
            return roi
        
if __name__ == "__main__":

    roi = detect_object()
    print("Washer detected at " + roi)