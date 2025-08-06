import requests
import os
import socket

import pathlib

USER = os.environ.get("PUSHOVER_USER_ID")
API = os.environ.get("PUSHOVER_API_TOKEN")


def send_notification(file: str, message: str) -> requests.Response | str:
    if USER is None:
        return "Failed to send notification; no environment variable 'PUSHOVER_USER_ID'."
    if API is None:
        return "Failed to send notification; no environment variable 'PUSHOVER_API_TOKEN'."
    payload = {"message": "{} on {}: {}".format(pathlib.Path(file).name, socket.gethostname(), message), "user": USER,
               "token": API}
    return requests.post('https://api.pushover.net/1/messages.json', data=payload, headers={'User-Agent': 'Python'})
