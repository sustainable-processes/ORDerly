import logging
from orderly.cleaning.cleaner import main

LOG = logging.getLogger(__name__)
logging.basicConfig(
    filename='cleaning.log', encoding='utf-8', 
    format='%(name)s - %(levelname)s - %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)

LOG.info("Running clean main")
main()
LOG.info("Completed clean main")
