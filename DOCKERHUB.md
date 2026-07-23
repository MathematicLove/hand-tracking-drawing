# Hand Tracking Drawing

Draw in the air with your hand, then view the drawing as a rotatable 3D object.
Hand tracking uses the MediaPipe Hand Landmarker. Rendering and the UI use OpenCV.

## Image

- Tag: flugmaschine/hand-tracking-drawing:latest
- Platform: linux/amd64
- Base: python:3.12-slim

## Run

This is a camera and GUI application. It needs a Linux host with a webcam and an
X11 display. It cannot reach the camera or screen on Docker Desktop for macOS or
Windows.

```
xhost +local:docker

docker run --rm \
  --device /dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --network host \
  flugmaschine/hand-tracking-drawing:latest
```

Saved drawings go to /app/output inside the container. Mount a folder to keep them:

```
  -v $(pwd)/output:/app/output
```

## Controls

- One index finger: draw
- Index and middle finger: move cursor without drawing
- Fist and hold: grab the drawing, move to rotate it in 3D
- Two fists: scale
- Open palm: release, back to drawing
- OK sign, hold: clear the canvas
- Pinky, hold: undo the last stroke
- Middle finger: shows :(( for a second
- Both palms waving: say goodbye, closes the app

Keys: S save, C clear, U undo, 1 to 6 color, [ and ] size, SPACE 2D or 3D,
R reset view, K skeleton, D debug, H help, Q quit.
