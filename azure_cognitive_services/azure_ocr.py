"""
Azure OCR: get characters from image url
"""

from io import BytesIO
import os
import time
import requests
from PIL import Image, ImageDraw, ImageFont
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials


SUBSCRIPTION_KEY = os.getenv("SUBSCRIPTION_KEY")
ENDPOINT = os.getenv("ENDPOINT")
CV_CLIENT = ComputerVisionClient(
    ENDPOINT, CognitiveServicesCredentials(SUBSCRIPTION_KEY)
)


def main():
    """
    Azure OCR: get characters from image url
    """
    url = "https://i.imgur.com/qyWiqQv.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    draw = ImageDraw.Draw(img)
    font_size = int(5e-2 * img.size[1])
    fnt = ImageFont.truetype("../static/TaipeiSansTCBeta-Regular.ttf", size=font_size)

    ocr_results = CV_CLIENT.read(url, raw=True)
    # Get the operation location (URL with an ID at the end) from the response
    operation_location_remote = ocr_results.headers["Operation-Location"]
    # Grab the ID from the URL
    operation_id = operation_location_remote.split("/")[-1]
    # Call the "GET" API and wait for it to retrieve the results
    while True:
        get_handw_text_results = CV_CLIENT.get_read_result(operation_id)
        if get_handw_text_results.status not in ["notStarted", "running"]:
            break
        time.sleep(1)

    # Get detected text
    if get_handw_text_results.status == OperationStatusCodes.succeeded:
        for text_result in get_handw_text_results.analyze_result.read_results:
            for line in text_result.lines:
                bounding_box = line.bounding_box
                bounding_box += bounding_box[:2]
                draw.line(
                    line.bounding_box, fill=(255, 0, 0), width=int(font_size / 10)
                )
                left = line.bounding_box[0]
                top = line.bounding_box[1]
                draw.text(
                    [left, top - font_size],
                    line.text,
                    fill=(0, 255, 255),
                    font=fnt,
                )
    # Filter text for Taiwan license plate
    img.save("output.png")
    print("Done!")
    print("Please check ouptut.png")


if __name__ == "__main__":
    main()
