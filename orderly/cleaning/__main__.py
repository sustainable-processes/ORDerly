import logging

from orderly.cleaning.cleaner import main

LOG = logging.getLogger(__name__)

LOG.info("Running clean main")
main()
LOG.info("Complete clean main")
