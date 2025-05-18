# AI Virtual Mouse 👆
- Control your computer's mouse cursor using hand gestures captured by your webcam. This project uses computer vision and hand tracking to create a touchless interface for your computer.
  
## ✨ Features

### Hand Gesture Control: 
- Move your cursor and click without touching any physical device
- Intuitive Gestures: Simple hand movements for mouse control

- Raise index finger to move the cursor
- Pinch index and middle fingers to click


**Smooth Movement**: Implemented cursor smoothing to reduce jitter
**Visual Feedback**: See your hand landmarks and active tracking area
**Customizable**: Easily adjust sensitivity and tracking parameters

### 🧠 How It Works
This project uses OpenCV and MediaPipe to detect and track hand landmarks in real-time. The system:

1. Captures video from your webcam
2. Detects hand landmarks (21 points on each hand)
3. Tracks specific finger positions
4. Maps hand coordinates to screen coordinates
5. Controls mouse cursor based on gestures

#### Core Implementation

- Hand Detection and Tracking
- python# Detect hand in the image
- img = detector.findHands(img)
- landmark_list, bounding_box = detector.findPosition(img)

#### Check which fingers are up

``` bash
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
``` 
#### Mouse Click with Pinch Gesture

```python Clicking Mode: Both index and middle fingers are up
if fingers[1] == 1 and fingers[2] == 1:
    # Find distance between fingertips
    length, img, line_info = detector.findDistance(8, 12, img)
    
    # Click mouse if fingers are pinched (close together)
    if length < 40:
        autopy.mouse.click()
```
#### 🛠️ Requirements

- Python 3.7+
- Webcam
- Libraries:

- OpenCV (opencv-python)
- MediaPipe (mediapipe)
- NumPy (numpy)
- AutoPy (autopy) - for mouse control



#### 📋 Installation

##### Clone this repository:
``` bash
git clone https://github.com/yourusername/ai-virtual-mouse.git
cd ai-virtual-mouse

Create and activate a virtual environment (recommended):
bashpython -m venv .venv
```
####  Windows
```
.venv\Scripts\activate
```
#### macOS/Linux
```
source .venv/bin/activate
```
**Install the required dependencies**:
``` bash
pip install opencv-python mediapipe numpy autopy
```

**🚀 Usage**
```
# Run the main script:
python AiVirtualMouseProject.py
```

- Position your hand within view of the webcam
= Use these gestures to control your mouse:

**Move Cursor**: Raise only your index finger
**Click**: Raise both index and middle fingers and bring them close together
**Exit Program**: Press 'q' on your keyboard



#### ⚙️ Configuration
You can customize the behavior by modifying the constants at the top of AiVirtualMouseProject.py:
python# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480



Made with ❤️ by Muhammad Hanzla
Feel free to star ⭐ this repository if you found it useful!
