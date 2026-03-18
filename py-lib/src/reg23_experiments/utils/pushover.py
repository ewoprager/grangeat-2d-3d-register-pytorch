import requests
import os
import socket
import logging

logger = logging.getLogger(__name__)

import pathlib

USER = os.environ.get("PUSHOVER_USER_ID")
API = os.environ.get("PUSHOVER_API_TOKEN")


def send_notification(file: str, message: str) -> bool:
    if USER is None:
        logger.error("Failed to send notification; no environment variable 'PUSHOVER_USER_ID'.")
        return False
    if API is None:
        logger.error("Failed to send notification; no environment variable 'PUSHOVER_API_TOKEN'.")
        return False
    payload = {"message": "{} on {}: {}".format(pathlib.Path(file).name, socket.gethostname(), message), "user": USER,
               "token": API}
    logger.info("Pushover post response: {}".format(
        requests.post('https://api.pushover.net/1/messages.json', data=payload, headers={'User-Agent': 'Python'})))
    return True
