"""
Azure object detection
"""
import os
from io import BytesIO
import requests
from PIL import Image, ImageDraw, ImageFont
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials


SUBSCRIPTION_KEY = os.getenv("SUBSCRIPTION_KEY")
ENDPOINT = os.getenv("ENDPOINT")
CV_CLIENT = ComputerVisionClient(
    ENDPOINT, CognitiveServicesCredentials(SUBSCRIPTION_KEY)
)


def main():
    """
    Azure object detection
    """
    url = "https://i.imgur.com/Js5H6Qa.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    draw = ImageDraw.Draw(img)
    font_size = int(5e-2 * img.size[1])
    fnt = ImageFont.truetype("../static/TaipeiSansTCBeta-Regular.ttf", size=font_size)

    object_detection = CV_CLIENT.detect_objects(url)
    if len(object_detection.objects) > 0:
        for obj in object_detection.objects:
            left = obj.rectangle.x
            top = obj.rectangle.y
            right = obj.rectangle.x + obj.rectangle.w
            bot = obj.rectangle.y + obj.rectangle.h
            name = obj.object_property
            confidence = obj.confidence
            print("{} at location {}, {}, {}, {}".format(name, left, right, top, bot))
            draw.rectangle([left, top, right, bot], outline=(255, 0, 0), width=3)
            draw.text(
                [left, top + font_size],
                "{0} {1:0.1f}".format(name, confidence * 100),
                fill=(255, 0, 0),
                font=fnt,
            )
    img.save("output.png")
    print("Done!")
    print("Please check ouptut.png")


if __name__ == "__main__":
    main()
