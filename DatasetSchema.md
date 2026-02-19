├── <OBJECT NAME>
│   ├── depth
│   ├── f_t.csv
│   ├── gelsight
│   ├── gripper.csv
│   ├── label.csv
│   ├── label_old.csv
│   ├── rgb
│   ├── robot.csv
│   ├── side_cam
│   ├── stages.csv
│   ├── top_cam
│   └── visualization.mp4

- <OBJECT NAME>: <OBJECT>_<TIMESTAMP>_<GRIPPER_FORCE>_<POSE_IDX>
    - depth: `.tif` images captured by the sensor. Captured at 4-5 Hz
    - `f_t.csv`: Columns with `time,Fx,Fy,Fz,Tx,Ty,Tz`. Captured at 70 Hz
    - gelsight: .JPG images captured at 22-26 Hz
    - gripper.csv: Columns: `timestamp,left,right` 10 Hz
    - side_cam: .JPG images at 30Hz
    - rgb: .JPG images at < 10Hz

- one sample: 
    - first 20 s from one experiment
    - get F1 visual images per second per camera
    - get F2 force readings per second
    - get F3 gripper readings per second
    - get F4 gelsight images per second
    - get F5 depth images per second

- Architecture
    - RGB
        - ResNet50 ImageNet100 pretrained weights
        - Output embedding vector size V1
    - GelSight
        - ResNet50 ImageNet100 pretrained weights
        - Output embedding vector size V4
    - Force
        - F2 x 6 vector
    - Gripper
        - F3 x 2 vector
    - Ignore depth for now
    - Ignore side-cam for now

    - Make big concat col vector - <Size>
    - FC <Size> x <EmbeddingSpace>
    - LSTM
        - In: <EmbeddingSpace>
        - <LSTM Out>: CellState, HiddenState 
        - Pass both CellState and HiddenState to FC for classification
    - FC
        - In: <LSTM Out>
        - Out: 0 or 1