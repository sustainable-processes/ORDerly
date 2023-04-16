import logging
from orderly.extract.main import main_click

# LOG = logging.getLogger(__name__)
# logging.basicConfig(
#     filename="extraction.log",
#     encoding="utf-8",
#     format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
#     datefmt="%d-%b-%y %H:%M:%S",
#     level=logging.INFO,
# )

# LOG.info("Running extract main")
main_click()
# LOG.info("Completed extract main")
