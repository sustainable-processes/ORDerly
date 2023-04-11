import logging
from orderly.clean.cleaner import main_click

LOG = logging.getLogger(__name__)
logging.basicConfig(
    # filename="cleaning.log",
    # encoding="utf-8",
    format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    # level=logging.INFO,
    level=logging.DEBUG,
)

LOG.info("Running clean main")
main_click()
LOG.info("Completed clean main")
