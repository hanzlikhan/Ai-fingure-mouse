import cv2
import mediapipe as mp
import math
import numpy as np

class HandGestureTracker:
    """A class for detecting and tracking hand gestures using MediaPipe."""
    
    # Landmark indices for finger tips (thumb, index, middle, ring, pinky)
    FINGER_TIP_IDS = [4, 8, 12, 16, 20]

    def __init__(self, static_mode: bool = False, max_hands: int = 2, 
                 detection_confidence: float = 0.5, tracking_confidence: float = 0.5):
        """
        Initialize the hand gesture tracker with MediaPipe Hands.

        Args:
            static_mode (bool): If True, treat each frame as a static image
            max_hands (int): Maximum number of hands to detect
            detection_confidence (float): Minimum confidence for hand detection
            tracking_confidence (float): Minimum confidence for landmark tracking
        """
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.landmark_list = []  # Store detected hand landmarks

    def detect_hands(self, frame: np.ndarray, draw_landmarks: bool = True) -> np.ndarray:
        """
        Detect hands in the frame and optionally draw landmarks.

        Args:
            frame (np.ndarray): Input BGR frame from camera
            draw_landmarks (bool): If True, draw hand landmarks and connections

        Returns:
            np.ndarray: Processed frame with optional drawings
        """
        # Convert BGR to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)

        # Draw hand landmarks if detected
        if self.results.multi_hand_landmarks and draw_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return frame

    def get_landmark_positions(self, frame: np.ndarray, hand_index: int = 0, 
                              draw_markers: bool = True) -> tuple[list, tuple]:
        """
        Get positions of hand landmarks and bounding box for a specified hand.

        Args:
            frame (np.ndarray): Input BGR frame
            hand_index (int): Index of the hand to process (default: 0)
            draw_markers (bool): If True, draw landmarks and bounding box

        Returns:
            tuple: List of landmarks ([id, x, y]) and bounding box (xmin, ymin, xmax, ymax)
        """
        self.landmark_list = []
        x_coords, y_coords = [], []
        bounding_box = []

        if self.results.multi_hand_landmarks:
            target_hand = self.results.multi_hand_landmarks[hand_index]
            frame_height, frame_width, _ = frame.shape

            # Extract landmark coordinates
            for idx, landmark in enumerate(target_hand.landmark):
                pixel_x = int(landmark.x * frame_width)
                pixel_y = int(landmark.y * frame_height)
                x_coords.append(pixel_x)
                y_coords.append(pixel_y)
                self.landmark_list.append([idx, pixel_x, pixel_y])

                # Draw a circle at each landmark
                if draw_markers:
                    cv2.circle(frame, (pixel_x, pixel_y), 5, (255, 0, 255), cv2.FILLED)

            # Calculate bounding box
            if x_coords and y_coords:
                xmin, xmax = min(x_coords), max(x_coords)
                ymin, ymax = min(y_coords), max(y_coords)
                bounding_box = (xmin, ymin, xmax, ymax)

                # Draw bounding box with padding
                if draw_markers:
                    cv2.rectangle(
                        frame,
                        (xmin - 20, ymin - 20),
                        (xmax + 20, ymax + 20),
                        (0, 255, 0),  # Green color
                        2
                    )

        return self.landmark_list, bounding_box

    def count_raised_fingers(self) -> list[int]:
        """
        Determine which fingers are raised based on landmark positions.

        Returns:
            list[int]: List of 1s (raised) and 0s (not raised) for [thumb, index, middle, ring, pinky]
        """
        raised_fingers = []

        if not self.landmark_list:
            return [0] * 5

        # Thumb: Check x-coordinate relative to its base
        thumb_tip_x = self.landmark_list[self.FINGER_TIP_IDS[0]][1]
        thumb_base_x = self.landmark_list[self.FINGER_TIP_IDS[0] - 1][1]
        raised_fingers.append(1 if thumb_tip_x > thumb_base_x else 0)

        # Other fingers: Check y-coordinate relative to their base
        for finger_idx in range(1, 5):
            tip_y = self.landmark_list[self.FINGER_TIP_IDS[finger_idx]][2]
            base_y = self.landmark_list[self.FINGER_TIP_IDS[finger_idx] - 2][2]
            raised_fingers.append(1 if tip_y < base_y else 0)

        return raised_fingers

    def measure_distance(self, point1_idx: int, point2_idx: int, frame: np.ndarray, 
                        draw_elements: bool = True, circle_radius: int = 15, 
                        line_thickness: int = 3) -> tuple[float, np.ndarray, list]:
        """
        Measure the distance between two landmarks and optionally draw the line and circles.

        Args:
            point1_idx (int): Index of the first landmark
            point2_idx (int): Index of the second landmark
            frame (np.ndarray): Input BGR frame
            draw_elements (bool): If True, draw line and circles
            circle_radius (int): Radius of drawn circles
            line_thickness (int): Thickness of drawn line

        Returns:
            tuple: Distance, modified frame, and coordinates [x1, y1, x2, y2, cx, cy]
        """
        x1, y1 = self.landmark_list[point1_idx][1:]
        x2, y2 = self.landmark_list[point2_idx][1:]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw line and circles if requested
        if draw_elements:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), line_thickness)
            cv2.circle(frame, (x1, y1), circle_radius, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), circle_radius, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (center_x, center_y), circle_radius, (0, 0, 255), cv2.FILLED)

        # Calculate Euclidean distance
        distance = math.hypot(x2 - x1, y2 - y1)
        return distance, frame, [x1, y1, x2, y2, center_x, center_y]

def main():
    """Demonstrate hand tracking with FPS display."""
    last_frame_time = 0
    camera = cv2.VideoCapture(0)
    gesture_tracker = HandGestureTracker()

    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to capture frame")
            break

        # Detect hands and get landmark positions
        frame = gesture_tracker.detect_hands(frame)
        landmarks, bounding_box = gesture_tracker.get_landmark_positions(frame)

        # Print thumb tip position if detected
        if landmarks:
            print(f"Thumb tip position: {landmarks[4]}")

        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - last_frame_time) if current_time != last_frame_time else 0
        last_frame_time = current_time

        cv2.putText(
            frame,
            f'FPS: {int(fps)}',
            (10, 70),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 255),  # Purple color
            3
        )

        # Display output
        cv2.imshow("Hand Gesture Tracker", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()