# Overview

This repository contains scripts for marker‑based hand‑movement tracking.

| Script | Description | Required Arguments |
| --- | --- | --- |
| [`calibration_zed.py`](calibration_zed.py) | Performs camera calibration using a checkerboard with multiple cameras. | `--calib_svo_path`, `--calib_output_path` |
| [`hsv_selector.py`](hsv_selector.py) | Finds the appropriate HSV limits for each marker and camera. | None |
| [`blob_detector.py`](blob_detector.py) | Contains the main tracking logic (see below). | See below |

## Requirements
- Python 3.10
- ZED SDK (and Python API): https://www.stereolabs.com/docs/development/python/install
- Project dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Calibration

Run ZED calibration to generate per‑camera YAML files:

```bash
python calibration_zed.py \
  --calib_svo_path ./data/calib_svo \
  --calib_output_path ./data/calib
```

Arguments:
- `--calib_svo_path` (str) — Path to a folder with calibration SVO files.
- `--calib_output_path` (str) — Output folder for calibration files.

# Hand tracking

The hand‑tracking logic is implemented in `blob_detector.py`.

## Inputs

```bash
python blob_detector.py \
  --project_name MyProject \
  --visuals \
  --original_path ./data/raw \
  --calib_path ./data/calib
```

- `--project_name` (str) — Name of the project.
- `--visuals` (flag) — Enable visualization during tracking (store_true).
- `--original_path` (str) — Path to the CSV and SVO files that contain the segments.
- `--calib_path` (str) — Path to the calibration files (from `calibration_zed.py`).

## Output JSON Structure

Each segment produces a JSON file with motion‑tracking data. Example:

```json
{
  "segment_id": "1",
  "start_timestamp": 0.0,
  "end_timestamp": 12.34,
  "robot": {
    "base_transform": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
    "states": [
      {
        "id": 0,
        "frame": 15,
        "finger_distance": 34.2,
        "goal_position": [[1,0,0,0.10],[0,1,0,0.20],[0,0,1,0.30],[0,0,0,1]]
      },
      {
        "id": 1,
        "frame": 16,
        "finger_distance": 36.1,
        "goal_position": [[1,0,0,0.10],[0,1,0,0.20],[0,0,1,0.30],[0,0,0,1]]
      }
    ]
  },
  "cameras": [
    {
      "camera_id": "3924863",
      "intrinsics": [700.1, 0.0, 640.0, 0.0, 700.1, 360.0, 0.0, 0.0, 1.0],
      "extrinsics": [[1,0,0,0.05],[0,1,0,0.00],[0,0,1,0.02],[0,0,0,1]],
      "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
      "d_fov": 87.0,
      "focal_length": 700.1,
      "h_fov": 80.0,
      "v_fov": 52.0
    }
  ]
}
```
