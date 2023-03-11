import ORDerly.cleaning.USPTO_cleaning
import ORDerly.extraction.USPTO_extraction

from ORDerly.cleaning.USPTO_cleaning import Cleaner
from ORDerly.extraction.USPTO_extraction import OrdToPickle

import logging

logging.getLogger("uspto_cleaning").addHandler(logging.NullHandler())
