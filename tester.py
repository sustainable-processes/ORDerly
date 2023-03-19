# %%
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("uspto_cleaning").propagate = True

LOG = logging.getLogger(__name__)

LOG.warning("Heloo")


import orderly


# %%

logging.getLogger("uspto_cleaning.main")


# %%
# %%

import importlib.resources

importlib.resources.as_file(orderly.data)

# %%
