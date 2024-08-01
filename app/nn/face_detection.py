import cv2
from cv2.typing import MatLike
from PIL import Image
from facenet_pytorch import MTCNN


mtcnn = MTCNN(keep_all=True)


def detect_face(frame: MatLike):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    boxes, probs = mtcnn.detect(pil_image)

    threshold = 0.9

    if boxes is not None:
        return boxes[probs > threshold]
    return []


def draw_bounding_box(frame: MatLike, bboxes):
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box)

    color = (50, 205, 50)
    face = cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    return face


def pipeline():
    capture = cv2.VideoCapture("data/FaceSwap_046_904.mp4")

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        bboxes = detect_face(frame)
        if bboxes is not None:
            frame = draw_bounding_box(frame, bboxes)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
