import logging
from orderly.extraction.main import main

LOG = logging.getLogger(__name__)
logging.basicConfig(
    filename="extraction.log",
    encoding="utf-8",
    format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)

LOG.info("Running extract main")
main()
LOG.info("Completed extract main")
