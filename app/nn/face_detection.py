import cv2
from cv2.typing import MatLike
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm


mtcnn = MTCNN(keep_all=True)


def detect_face(frame: MatLike):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    boxes, probs = mtcnn.detect(pil_image)

    threshold = 0.9

    if boxes is not None:
        return pil_image, boxes[probs > threshold]
    return pil_image, []


def crop_frame_to_face(frame: Image.Image, bounding_boxes: list[int]):
    for box in bounding_boxes:
        x1, y1, x2, y2 = map(int, box)

    face = frame.crop((x1, y1, x2, y2))
    return face


def setup_model():
    model = timm.create_model(
        "hf_hub:timm/xception41.tf_in1k",
        pretrained=True,
        num_classes=2,
    )

    model.load_state_dict(
        torch.load("./models/Celeb-DF-v2-Split10-XceptionNetBaseline/model.pth")
    )

    return model


def inference(face):
    transformer = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ]
    )

    image_tensor = transformer(face)
    image_tensor = image_tensor.unsqueeze(0).to("cuda")

    model = setup_model()
    model.to("cuda")
    model.eval()

    with torch.no_grad():
        prediction = model(image_tensor)

    probabilities = F.softmax(prediction, dim=1)
    predicted_class_index = torch.argmax(probabilities, dim=1).item()

    classes = ["FAKE", "REAL"]
    predicted_class_name = classes[predicted_class_index]
    return predicted_class_name


def draw_bounding_box(frame: MatLike, bboxes, predicted_class):
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box)

    color = (50, 205, 50)
    frame = cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    frame = cv2.putText(
        frame,
        predicted_class,
        (x1, y2 + 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=color,
        thickness=5,
    )

    return frame


def pipeline():
    capture = cv2.VideoCapture("data/FaceSwap_046_904.mp4")

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        pil_image, bboxes = detect_face(frame)
        if bboxes is not None:
            cropped_face = crop_frame_to_face(pil_image, bboxes)
            predicted_class = inference(cropped_face)
            frame = draw_bounding_box(frame, bboxes, predicted_class)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
