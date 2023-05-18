import os
import socket
from datetime import datetime


def log_dir(prefix="", comment=""):
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(
        "runs", prefix + current_time + "_" + socket.gethostname() + comment
    )
    return log_dir
