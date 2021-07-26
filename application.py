"""
Object detection and image description on LINE bot
"""
from datetime import datetime, timezone, timedelta
import os
import json
import time
import requests
from flask import Flask, request, abort
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.face import FaceClient
from azure.storage.blob import BlobServiceClient
from azure.cognitiveservices.speech import (
    SpeechConfig,
    SpeechSynthesizer,
)
from azure.cognitiveservices.speech.audio import AudioOutputConfig

from msrest.authentication import CognitiveServicesCredentials
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
    FlexSendMessage,
    ImageMessage,
)
from pymongo import MongoClient
from PIL import Image, ImageDraw, ImageFont
import investpy

app = Flask(__name__)


CONFIG = json.load(open("/home/config.json", "r"))

SUBSCRIPTION_KEY = CONFIG["azure"]["subscription_key"]
ENDPOINT = CONFIG["azure"]["endpoint"]
CV_CLIENT = ComputerVisionClient(
    ENDPOINT, CognitiveServicesCredentials(SUBSCRIPTION_KEY)
)

FACE_KEY = CONFIG["azure"]["face_key"]
FACE_END = CONFIG["azure"]["face_end"]
FACE_CLIENT = FaceClient(FACE_END, CognitiveServicesCredentials(FACE_KEY))
PERSON_GROUP_ID = "triathlon"


MONGO = MongoClient(CONFIG["azure"]["mongo_uri"], retryWrites=False)
DB = MONGO["face_register"]

ML_URL = CONFIG["azure"]["azureml_endpoint"]

CONNECT_STR = CONFIG["azure"]["blob_connect"]
CONTAINER = CONFIG["azure"]["blob_container"]
BLOB_SERVICE = BlobServiceClient.from_connection_string(CONNECT_STR)

TRANS_KEY = CONFIG["azure"]["trans_key"]
SPEECH_KEY = CONFIG["azure"]["speech_key"]

LINE_SECRET = CONFIG["line"]["line_secret"]
LINE_TOKEN = CONFIG["line"]["line_token"]
LINE_BOT = LineBotApi(LINE_TOKEN)
HANDLER = WebhookHandler(LINE_SECRET)


@app.route("/")
def hello():
    "hello world"
    return "Hello World!!!!!"


def check_registered(name):
    collect_register = DB["line"]
    return collect_register.find_one({"name": name})


def face_login(name, user_id):
    result = check_registered(name)
    if result:
        if result["userId"] == user_id:
            collect_login = DB["daily_login"]
            now = datetime.now()
            post = {"userId": user_id, "time": now.timestamp()}
            collect_login.insert_one(post)


def is_login(user_id):
    collect_login = DB["daily_login"]
    yesterday = datetime.now() - timedelta(days=1)
    result = collect_login.count_documents(
        {"$and": [{"userId": user_id}, {"time": {"$gte": yesterday.timestamp()}}]}
    )

    return result > 0


def upload_blob(container, path):
    blob_client = BLOB_SERVICE.get_blob_client(container=container, blob=path)
    with open(path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    data.close()
    return blob_client.url


def azure_describe(url):
    """
    Output azure image description result
    """
    description_results = CV_CLIENT.describe_image(url)
    output = ""
    for caption in description_results.captions:
        output += "'{}' with confidence {:.2f}% \n".format(
            caption.text, caption.confidence * 100
        )
    return output


def azure_ocr(url):
    """
    Azure OCR: get characters from image url
    """
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
    text = []
    if get_handw_text_results.status == OperationStatusCodes.succeeded:
        for text_result in get_handw_text_results.analyze_result.read_results:
            for line in text_result.lines:
                text.append(line.text)
    return text


def azure_speech(string, message_id):
    speech_config = SpeechConfig(subscription=SPEECH_KEY, region="eastus2")
    speech_config.speech_synthesis_language = "ko-KR"
    audio_config = AudioOutputConfig(filename="{}.wav".format(message_id))

    synthesizer = SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )
    synthesizer.speak_text_async(string)
    link = upload_blob(CONTAINER, "{}.wav".format(message_id))
    output = {
        "type": "button",
        "flex": 2,
        "style": "primary",
        "color": "#1E90FF",
        "action": {"type": "uri", "label": "Voice", "uri": link},
        "height": "sm",
    }
    return output


def azure_translation(string, message_id):
    trans_url = "https://api.cognitive.microsofttranslator.com/translate"

    params = {"api-version": "2.0", "to": ["zh-Hant"]}

    headers = {
        "Ocp-Apim-Subscription-Key": TRANS_KEY,
        "Content-type": "application/json",
        "Ocp-Apim-Subscription-Region": "eastus2",
    }

    # You can pass more than one object in body.
    body = [{"text": string}]

    req = requests.post(trans_url, params=params, headers=headers, json=body)
    response = req.json()
    output = ""
    speech_button = ""
    ans = []
    for i in response:
        ans.append(i["translations"][0]["text"])
    language = response[0]["detectedLanguage"]["language"]
    if language == "ko":
        output = " ".join(string) + "\n" + " ".join(ans)
        speech_button = azure_speech(string, message_id)
    return output, speech_button


def azure_object_detection(url, filename):
    img = Image.open(filename)
    draw = ImageDraw.Draw(img)
    font_size = int(5e-2 * img.size[1])
    fnt = ImageFont.truetype("static/TaipeiSansTCBeta-Regular.ttf", size=font_size)
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
                "{} {}".format(name, confidence),
                fill=(255, 0, 0),
                font=fnt,
            )
    img.save(filename)
    # image = IMGUR_CLIENT.image_upload(filename, "", "")
    # link = image["response"]["data"]["link"]
    link = upload_blob(CONTAINER, filename)
    os.remove(filename)
    return link


def azure_face_recognition(filename):
    """
    Azure face recognition
    """
    img = open(filename, "r+b")
    detected_face = FACE_CLIENT.face.detect_with_stream(
        img, detection_model="detection_01"
    )
    if len(detected_face) != 1:
        return ""
    results = FACE_CLIENT.face.identify([detected_face[0].face_id], PERSON_GROUP_ID)
    if len(results) == 0:
        return "unknown"
    result = results[0].as_dict()
    if len(result["candidates"]) == 0:
        return "unknown"
    if result["candidates"][0]["confidence"] < 0.5:
        return "unknown"
    person = FACE_CLIENT.person_group_person.get(
        PERSON_GROUP_ID, result["candidates"][0]["person_id"]
    )
    return person.name


@app.route("/callback", methods=["POST"])
def callback():
    """
    LINE bot webhook callback
    """
    # get X-Line-Signature header value
    signature = request.headers["X-Line-Signature"]
    print(signature)
    body = request.get_data(as_text=True)
    print(body)
    try:
        HANDLER.handle(body, signature)
    except InvalidSignatureError:
        print(
            "Invalid signature. Please check your channel access token/channel secret."
        )
        abort(400)
    return "OK"


@HANDLER.add(MessageEvent, message=TextMessage)
def handle_message(event):
    """
    Reply text message
    """
    if event.message.text == "currency":
        recent = investpy.get_currency_cross_recent_data("USD/TWD")
        message = TextSendMessage(text=recent.Close.values[-1])
    elif event.message.text == "prediction" & is_login(event.source.user_id):
        recent = investpy.get_currency_cross_recent_data("USD/TWD")
        data = {"data": ""}
        input_data = json.dumps(data)
        headers = {"Content-Type": "application/json"}
        resp = requests.post(ML_URL, input_data, headers=headers)
        message = TextSendMessage(
            text="now: {}, prediction: {}".format(recent.Close.values[-1], resp.text)
        )
    else:
        message = TextSendMessage(text=event.message.text)
    LINE_BOT.reply_message(event.reply_token, message)


@HANDLER.add(MessageEvent, message=ImageMessage)
def handle_content_message(event):
    """
    Reply Image message with results of image description and objection detection
    """
    print(event.message)
    print(event.source.user_id)
    print(event.message.id)

    with open("templates/detect_result.json", "r") as f_r:
        bubble = json.load(f_r)
    f_r.close()
    filename = "{}.jpg".format(event.message.id)
    message_content = LINE_BOT.get_message_content(event.message.id)
    with open(filename, "wb") as f_w:
        for chunk in message_content.iter_content():
            f_w.write(chunk)
    f_w.close()
    img = Image.open(filename)
    link = upload_blob(CONTAINER, filename)
    name = azure_face_recognition(filename)
    output = ""
    if name != "":
        now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
        output = "{0}, {1}".format(name, now)
        face_login(name, event.source.user_id)
    else:
        text = azure_ocr(link)
        if len(text) > 0:
            output, speech_button = azure_translation(" ".join(text), event.message.id)
            bubble["body"]["contents"].append(speech_button)
        if output == "":
            link_ob = azure_object_detection(link, filename)
            output = azure_describe(link)
            link = link_ob

    bubble["body"]["contents"][0]["text"] = output
    bubble["header"]["contents"][0]["url"] = link
    bubble["header"]["contents"][0]["aspectRatio"] = "{}:{}".format(
        img.size[0], img.size[1]
    )
    LINE_BOT.reply_message(
        event.reply_token, [FlexSendMessage(alt_text="Report", contents=bubble)]
    )
