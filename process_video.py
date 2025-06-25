import cv2
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm

def process_video(in_path: str, out_path: str) -> None:
    """
    Process a video: detect people, draw bounding boxes and save the result.

    The function uses [YOLOv11](https://docs.ultralytics.com/models/yolo11/) model to
    detect people in the video and draw bounding boxes around them.

    Parameters
    ----------
    in_path : str
        Path to the input video.
    out_path : str
        Path to the output video.

    Returns
    -------
    None

    Example Usage
    -------------
    ```python
    process_video('input.mp4', 'output.mp4')
    ```
    """
    model = YOLO('yolo11x.pt')
    
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise IOError(f'Cannot open {in_path}')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, classes=[0], verbose=False)

        for box in results[0].boxes:
            confidence = float(box.conf[0])
            if confidence > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f'Person: {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
