import orderly.clean.cleaner
from orderly.clean.cleaner import main as clean

import orderly.extract

import logging

logging.getLogger("uspto_cleaning").addHandler(logging.NullHandler())
