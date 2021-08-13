"""
Create a person and train on Azure.
"""
import glob
import os
import sys
import time
import argparse
import json
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import (
    TrainingStatusType,
    APIErrorException,
)

CONFIG = json.load(open("config.json", "r"))
FACE_KEY = CONFIG["azure"]["face_key"]
FACE_END = CONFIG["azure"]["face_end"]
FACE_CLIENT = FaceClient(FACE_END, CognitiveServicesCredentials(FACE_KEY))

# Create empty Person Group. Person Group ID must be lower case, alphanumeric, and/or with '-', '_'.
PERSON_GROUP_ID = "tibame"


def train_person(group_id, name, image_list):
    """
    Train Person
    """
    # Create a new person
    new = FACE_CLIENT.person_group_person.create(group_id, name)

    # Add image for the new person
    for image_file in image_list:
        img = open(image_file, "r+b")
        FACE_CLIENT.person_group_person.add_face_from_stream(
            group_id, new.person_id, img
        )

    # Training
    FACE_CLIENT.person_group.train(group_id)
    while True:
        training_status = FACE_CLIENT.person_group.get_training_status(group_id)
        print("Training status: {}.".format(training_status.status))
        print()
        if training_status.status is TrainingStatusType.succeeded:
            break
        elif training_status.status is TrainingStatusType.failed:
            sys.exit("Training the person group has failed.")
        time.sleep(5)


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g", "--group", help="person group", type=str, default="tibame"
    )
    parser.add_argument("-n", "--name", help="person's name", type=str)
    parser.add_argument("-i", "--image", help="image folder", type=str)
    args = parser.parse_args()
    return args


def main():
    """
    Create a person and train
    """
    args = parse_args()
    try:
        FACE_CLIENT.person_group.get(args.group)
        print("Person Group {} is existing.".format(args.group))
    except APIErrorException:
        FACE_CLIENT.person_group.create(person_group_id=args.group, name=args.group)
        print("Creat Person Group {}".format(args.group))
    files = glob.glob(os.path.join(args.image, "*"))
    train_person(group_id=args.group, name=args.name, image_list=files)


if __name__ == "__main__":
    main()
