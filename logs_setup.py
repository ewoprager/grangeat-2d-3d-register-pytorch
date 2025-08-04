from datetime import datetime
import socket
import sys
import logging.config

import pathlib


def setup_logger():
    hostname = socket.gethostname()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_full_path = pathlib.Path(sys.argv[0])
    logs_path = pathlib.Path("Logs/{}_{}_{}.log".format(hostname, timestamp, script_full_path.stem))

    if logs_path.is_file():
        logs_path = "{}_1".format(logs_path)

    header_string = "Hostname: {}\nTimestamp: {}\nCommand: ".format(hostname, timestamp)
    for arg in sys.argv:
        header_string = "{}{} ".format(header_string, arg)

    with open(logs_path, 'w') as file:
        file.write("{}\n\n".format(header_string))

    logging.config.fileConfig("logging.conf", disable_existing_loggers=False, defaults={"logs_path": str(logs_path)})
    # redirect warnings to logging
    logging.captureWarnings(True)
    logging.getLogger("py.warnings").setLevel(logging.WARNING)
    return logging.getLogger("radonRegistration")
