import logging

from orderly.extraction.main import main

LOG = logging.getLogger(__name__)
logging.basicConfig(
    filename='log/extraction.log', encoding='utf-8', 
    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
    level=logging.WARNING
)

logging.getLogger("orderly.extraction").propagate = True
logging.getLogger("orderly.extraction.extactor").propagate = True

LOG.info("Running extract main")
main()
LOG.info("Completed extract main")

