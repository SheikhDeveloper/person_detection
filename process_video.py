import cv2
from ultralytics import YOLO
from tqdm import tqdm
import torch


def process_video(in_path: str, out_path: str, class_label: int = 0, class_str: str = 'Person') -> None:
    """
    Process a video: detect objects of specified class, draw bounding boxes and save the result.

    The function uses [YOLOv11](https://docs.ultralytics.com/models/yolo11/) model to
    detect objects in the video and draw bounding boxes around them.

    Parameters
    ----------
    in_path : str
        Path to the input video.
    out_path : str
        Path to the output video.
    class_label : int
        Class label to detect(default=0, e.g. person).
    class_str : str
        Class string to display(default='Person').

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

        results = model.predict(
            frame,
            classes=[class_label],
            conf=0.5,
            device='cuda:0' if torch.cuda.is_available() else 'cpu')

        for box in results[class_label].boxes:
            confidence = box.conf[class_label]
            x1, y1, x2, y2 = map(int, box.xyxy[class_label])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f'{class_str} {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
