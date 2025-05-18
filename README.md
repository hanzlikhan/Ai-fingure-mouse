AI Virtual Mouse 👆
Control your computer's mouse cursor using hand gestures captured by your webcam. This project uses computer vision and hand tracking to create a touchless interface for your computer.
Show Image
✨ Features

Hand Gesture Control: Move your cursor and click without touching any physical device
Intuitive Gestures: Simple hand movements for mouse control

Raise index finger to move the cursor
Pinch index and middle fingers to click


Smooth Movement: Implemented cursor smoothing to reduce jitter
Visual Feedback: See your hand landmarks and active tracking area
Customizable: Easily adjust sensitivity and tracking parameters

🧠 How It Works
This project uses OpenCV and MediaPipe to detect and track hand landmarks in real-time. The system:

Captures video from your webcam
Detects hand landmarks (21 points on each hand)
Tracks specific finger positions
Maps hand coordinates to screen coordinates
Controls mouse cursor based on gestures

Core Implementation
Hand Detection and Tracking
python# Detect hand in the image
img = detector.findHands(img)
landmark_list, bounding_box = detector.findPosition(img)

# Check which fingers are up
fingers = detector.fingersUp()
Mouse Movement with Index Finger
python# Moving Mode: Only index finger is up
if fingers[1] == 1 and fingers[2] == 0:
    # Convert coordinates from webcam space to screen space
    mouse_x = np.interp(
        index_x, 
        (FRAME_REDUCTION, CAMERA_WIDTH - FRAME_REDUCTION), 
        (0, screen_width)
    )
    mouse_y = np.interp(
        index_y, 
        (FRAME_REDUCTION, CAMERA_HEIGHT - FRAME_REDUCTION), 
        (0, screen_height)
    )
    
    # Smoothen values to reduce jitter
    curr_x = prev_x + (mouse_x - prev_x) / SMOOTHENING
    curr_y = prev_y + (mouse_y - prev_y) / SMOOTHENING

    # Move mouse cursor
    autopy.mouse.move(screen_width - curr_x, curr_y)
Mouse Click with Pinch Gesture
python# Clicking Mode: Both index and middle fingers are up
if fingers[1] == 1 and fingers[2] == 1:
    # Find distance between fingertips
    length, img, line_info = detector.findDistance(8, 12, img)
    
    # Click mouse if fingers are pinched (close together)
    if length < 40:
        autopy.mouse.click()
🛠️ Requirements

Python 3.7+
Webcam
Libraries:

OpenCV (opencv-python)
MediaPipe (mediapipe)
NumPy (numpy)
AutoPy (autopy) - for mouse control



📋 Installation

Clone this repository:
bashgit clone https://github.com/yourusername/ai-virtual-mouse.git
cd ai-virtual-mouse

Create and activate a virtual environment (recommended):
bashpython -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

Install the required dependencies:
bashpip install opencv-python mediapipe numpy autopy


🚀 Usage

Run the main script:
bashpython AiVirtualMouseProject.py

Position your hand within view of the webcam
Use these gestures to control your mouse:

Move Cursor: Raise only your index finger
Click: Raise both index and middle fingers and bring them close together
Exit Program: Press 'q' on your keyboard



⚙️ Configuration
You can customize the behavior by modifying the constants at the top of AiVirtualMouseProject.py:
python# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Frame reduction (creates a "dead zone" around edges of the frame)
# Larger value means smaller active area
FRAME_REDUCTION = 100

# Smoothening factor for mouse movement
# Higher value = smoother but more delayed movement
SMOOTHENING = 7
Adjusting Hand Detection Parameters
You can also modify hand detection sensitivity in the HandTrackingModule.py file:
python# Create the hand detector with custom parameters
detector = htm.handDetector(
    mode=False,             # Set to True for static images
    maxHands=1,             # Maximum number of hands to detect
    detectionCon=0.7,       # Minimum detection confidence threshold
    trackCon=0.5            # Minimum tracking confidence threshold
)
📁 Project Structure

AiVirtualMouseProject.py: Main application script
HandTrackingModule.py: Hand detection and tracking module
README.md: Project documentation

Key Components
HandTrackingModule.py
pythonclass handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # Initialize MediaPipe Hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]  # Fingertip landmark IDs
        
    def fingersUp(self):
        fingers = []
        
        # Thumb: Check if thumb tip is to the right of the thumb base
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers: Check if fingertip is above the middle joint
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)  # Finger is up
            else:
                fingers.append(0)  # Finger is down
                
        return fingers
Main Application Flow
python# Main loop
while True:
    # Capture frame from webcam
    success, img = cap.read()
    
    # Find hand and landmarks
    img = detector.findHands(img)
    landmark_list, bounding_box = detector.findPosition(img)
    
    if landmark_list:
        # Process hand gestures for mouse control
        fingers = detector.fingersUp()
        
        # [Movement and click logic]
    
    # Display image with visual feedback
    cv2.imshow("AI Virtual Mouse", img)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
🔍 How to Improve

Add more gestures (right-click, scroll, drag)
Implement gesture recording and customization
Add support for both hands with different functions
Create a GUI for adjusting settings
Improve performance on lower-end hardware

Example: Adding Right-Click Functionality
You could extend the code to implement right-click functionality like this:
python# Right-Click: Index, middle and ring fingers are up
if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
    # Find distance between index and middle fingertips
    length, img, line_info = detector.findDistance(8, 12, img)
    
    # Right-click if fingers are close together
    if length < 40:
        cv2.circle(
            img, 
            (line_info[4], line_info[5]),
            15, (0, 0, 255), cv2.FILLED
        )
        
        # Perform right-click
        try:
            autopy.mouse.click(autopy.mouse.Button.RIGHT)
            time.sleep(0.3)  # Longer delay to prevent accidental clicks
        except Exception as e:
            print(f"Mouse right-click error: {e}")
🐛 Troubleshooting
No webcam detected
python# Try changing the camera index (from 0 to 1)
cap = cv2.VideoCapture(1)  # Try external webcam

# Or iterate through indices to find available cameras
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Found camera at index {i}")
        break
Hand detection issues

Ensure good lighting conditions
Keep your hand within the camera's field of view
Try adjusting the detection confidence threshold:

python# Increase detection confidence for more reliable but less sensitive detection
detector = htm.handDetector(detectionCon=0.7, trackCon=0.7)
Cursor movement is jerky
python# Increase smoothening value in AiVirtualMouseProject.py
SMOOTHENING = 15  # Higher value = smoother but more delayed movement

# Or implement a different smoothing algorithm
def exponential_smoothing(current, previous, alpha=0.3):
    return alpha * current + (1 - alpha) * previous
    
# Then use it in the movement code
curr_x = exponential_smoothing(mouse_x, prev_x)
curr_y = exponential_smoothing(mouse_y, prev_y)
📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgements

MediaPipe for the hand tracking solution
OpenCV for computer vision capabilities
AutoPy for mouse control functionality


Made with ❤️ by [Your Name]
Feel free to star ⭐ this repository if you found it useful!
