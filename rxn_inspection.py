# %%
import orderly

orderly.data.get_path_of_test_ords()

files = orderly.extract.main.get_file_names(
    directory=orderly.data.get_path_of_test_ords(),
    file_ending="00005539a1e04c809a9a78647bea649c.pb.gz",
)

from ord_schema import message_helpers as ord_message_helpers
from ord_schema.proto import dataset_pb2 as ord_dataset_pb2
import ord_schema.proto.reaction_pb2 as ord_reaction_pb2


# data = ord_message_helpers.load_message(str(files[0]), ord_dataset_pb2.Dataset)

# import orderly.extract.extractor

orderly.extract.extractor.OrdExtractor.load_data(files[0])

# %%
