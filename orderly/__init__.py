import orderly.cleaning.cleaner
from orderly.cleaning.cleaner import main as clean

import orderly.extraction

import logging

logging.getLogger("uspto_cleaning").addHandler(logging.NullHandler())
