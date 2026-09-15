# AI Virtual Whiteboard

A hand-tracking whiteboard built with Python, OpenCV, Pillow, and MediaPipe. It lets you draw on a virtual canvas using your webcam and hand gestures.

## Features

- Draw with index finger
- Erase with a fist/eraser gesture
- Move the cursor without drawing
- Undo and clear the canvas
- Change colors with a thumb + index pinch gesture
- Save the current drawing as an image

## Requirements

- Python 3.12
- Webcam

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   python whiteboard.py
   ```

## Controls

- `q` : quit
- `s` : save canvas

## Notes

This project expects a compatible MediaPipe/OpenCV stack. The pinned versions in `requirements.txt` are chosen to match the app’s import API.
