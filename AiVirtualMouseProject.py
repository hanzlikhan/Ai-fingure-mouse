import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

# Configuration constants
CAMERA_WIDTH, CAMERA_HEIGHT = 1000, 800  # Camera resolution
BOUNDARY_MARGIN = 100  # Margin for active area boundary
SMOOTH_FACTOR = 7  # Smoothing factor for mouse movement

# Mouse position tracking
prev_mouse_x, prev_mouse_y = 0, 0
curr_mouse_x, curr_mouse_y = 0, 0

# Frame rate tracking
last_frame_time = 0

def initialize_camera(width: int, height: int) -> cv2.VideoCapture:
    """
    Initialize the webcam with specified resolution.
    
    Args:
        width (int): Camera frame width
        height (int): Camera frame height
    
    Returns:
        cv2.VideoCapture: Configured camera object
    """
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return camera

def main():
    # Initialize camera and hand detector
    camera = initialize_camera(CAMERA_WIDTH, CAMERA_HEIGHT)
    hand_detector = htm.handDetector(maxHands=1)
    
    # Get screen dimensions
    screen_width, screen_height = autopy.screen.size()
    print(f"Screen Dimensions: {screen_width}x{screen_height}")
    
    global last_frame_time, prev_mouse_x, prev_mouse_y, curr_mouse_x, curr_mouse_y
    
    while True:
        # Capture and process frame
        success, frame = camera.read()
        if not success:
            print("Failed to capture frame")
            break
        
        # Detect hands and landmarks
        frame = hand_detector.findHands(frame)
        landmarks, bounding_box = hand_detector.findPosition(frame)
        
        if landmarks:
            # Extract index and middle finger tip coordinates
            index_finger_x, index_finger_y = landmarks[8][1:]  # Index finger tip
            middle_finger_x, middle_finger_y = landmarks[12][1:]  # Middle finger tip
            
            # Determine which fingers are raised
            raised_fingers = hand_detector.fingersUp()
            
            # Draw active area boundary
            cv2.rectangle(
                frame,
                (BOUNDARY_MARGIN, BOUNDARY_MARGIN),
                (CAMERA_WIDTH - BOUNDARY_MARGIN, CAMERA_HEIGHT - BOUNDARY_MARGIN),
                (255, 0, 255),  # Purple color
                2
            )
            
            # Moving Mode: Only index finger raised
            if raised_fingers[1] == 1 and raised_fingers[2] == 0:
                # Map camera coordinates to screen coordinates
                screen_x = np.interp(
                    index_finger_x,
                    (BOUNDARY_MARGIN, CAMERA_WIDTH - BOUNDARY_MARGIN),
                    (0, screen_width)
                )
                screen_y = np.interp(
                    index_finger_y,
                    (BOUNDARY_MARGIN, CAMERA_HEIGHT - BOUNDARY_MARGIN),
                    (0, screen_height)
                )
                
                # Smooth mouse movement
                curr_mouse_x = prev_mouse_x + (screen_x - prev_mouse_x) / SMOOTH_FACTOR
                curr_mouse_y = prev_mouse_y + (screen_y - prev_mouse_y) / SMOOTH_FACTOR
                
                # Move mouse (flip x-coordinate for natural movement)
                autopy.mouse.move(screen_width - curr_mouse_x, curr_mouse_y)
                
                # Visualize cursor position
                cv2.circle(
                    frame,
                    (index_finger_x, index_finger_y),
                    15,
                    (255, 0, 255),  # Purple color
                    cv2.FILLED
                )
                
                # Update previous mouse position
                prev_mouse_x, prev_mouse_y = curr_mouse_x, curr_mouse_y
            
            # Clicking Mode: Both index and middle fingers raised
            if raised_fingers[1] == 1 and raised_fingers[2] == 1:
                # Calculate distance between index and middle fingers
                distance, frame, line_info = hand_detector.findDistance(8, 12, frame)
                
                # Perform click if fingers are close
                if distance < 40:
                    cv2.circle(
                        frame,
                        (line_info[4], line_info[5]),
                        15,
                        (0, 255, 0),  # Green color
                        cv2.FILLED
                    )
                    autopy.mouse.click()
        
        # Calculate and display frame rate
        current_time = time.time()
        fps = 1 / (current_time - last_frame_time) if current_time != last_frame_time else 0
        last_frame_time = current_time
        
        cv2.putText(
            frame,
            f'FPS: {int(fps)}',
            (20, 50),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 0),  # Blue color
            3
        )
        
        # Display output window
        cv2.imshow("Virtual Mouse", frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()