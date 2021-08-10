"""
Predict mnist data with Azure machine learning
"""
import argparse
import json
import requests


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--endpoint_url", type=str, help="Endpoint url")
    args = parser.parse_args()
    return args


def main():
    """
    Predict mnist data with Azure machine learning
    """
    args = parse_args()
    data = {"data": ""}
    # Convert to JSON string
    input_data = json.dumps(data)
    # Set the content type
    headers = {"Content-Type": "application/json"}

    # Make the request and display the response
    resp = requests.post(args.endpoint_url, input_data, headers=headers)

    print("The answer is {}".format(resp.text))


if __name__ == "__main__":
    main()
