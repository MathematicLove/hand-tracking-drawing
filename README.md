# Hand Tracking Drawing

Draw in the air with your hand, then view the drawing as a rotatable 3D object.

<p align="center">
  <img src="https://github.com/MathematicLove/hand-tracking-drawing/blob/master/example/EXAMPLE_2.png"
       width="400"
       height="300"
       alt="Fig.1: Example of drawing">
</p>

**[DOCKER IMAGE HERE](https://hub.docker.com/r/flugmaschine/hand-tracking-drawing)**

## Model

Uses the **MediaPipe Hand Landmarker** (`models/hand_landmarker.task`), which returns
21 keypoints per hand. Gestures are classified from ratios and joint angles between
those keypoints, so recognition is independent of hand size, distance, or handedness.

## Logic

- **One index finger** - draw
- **Index + middle** - move cursor without drawing
- **Fist (hold)** - grab the drawing; move the fist to rotate it in 3D
- **Two fists** - scale (canvas in 2D, object in 3D)
- **Open palm** - release, back to drawing

Keys: `S` save, `C` clear, `U` undo, `1..6` color, `[` / `]` thickness,
`SPACE` 2D/3D, `R` reset view, `K` skeleton, `D` debug, `H` help, `Q` quit.

## Run

```bash
pip install -r requirements.txt
python src/main.py
```
