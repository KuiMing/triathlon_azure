import os
import json
import requests
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials


FACE_KEY = os.getenv("FACE_KEY")

# This endpoint will be used in all examples in this quickstart.
FACE_END = os.getenv("FACE_END")
FACE_CLIENT = FaceClient(FACE_END, CognitiveServicesCredentials(FACE_KEY))


def main():

    url = "https://i.imgur.com/rgNgeTC.jpg"

    detected_faces = FACE_CLIENT.face.detect_with_url(
        url=url,
        detectionModel="detection_02",
        return_recognition_model=True,
        return_face_landmarks=True,
        return_face_attributes=[
            "age",
            "blur",
            "gender",
            "headPose",
            "smile",
            "facialHair",
            "glasses",
            "emotion",
            "exposure",
        ],
    )
    face = detected_faces[0]
    rectangle = face.face_rectangle.as_dict()
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    draw = ImageDraw.Draw(img)
    bbox = [
        rectangle["left"],
        rectangle["top"],
        rectangle["left"] + rectangle["width"],
        rectangle["top"] + rectangle["height"],
    ]
    draw.rectangle(
        bbox,
        width=3,
        outline=(255, 0, 0),
    )
    img.show()
    print(json.dumps(face.face_attributes.as_dict(), indent=4))
    landmark = face.face_landmarks.as_dict()
    for i in landmark.values():
        draw.ellipse([i["x"] - 2, i["y"] - 2, i["x"] + 2, i["y"] + 2], fill=(255, 0, 0))
    img.crop(bbox).save("face_2.png")


if __name__ == "__main__":
    main()
