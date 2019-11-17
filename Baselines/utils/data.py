
import os, sys

sys.dont_write_bytecode = True
sys.path.append('/afs/crc.nd.edu/user/y/yzhang46/_DLResources/Datasets/')
from datasethandler import DataSetHandler
from models import ModelsV1


# class TinyImageNet:
#     def __init__(self):
#         self.model_handler = ModelHandler()
#         self.classification_structure = (500,'relu','dropout', 15, 'sigmoid')
#     def initialize_model(self, name):
#         model = self.model_handler.initialize_model(name,
#                                          cls_fmt=self.classification_structure)
#         return model
#     def load_model(self, model, pth_file):
#         """ Loads custom pre-trained weights into model.
#             model: either name or actual pytorch model  """
#         if isinstance(model, str):
#             model = self.model_handler.initialize_model(model,
#                                          cls_fmt=self.classification_structure)
#         state_dict = torch.load(pth_file)
#         if 'model_state_dict' in state_dict:
#             state_dict = state_dict['model_state_dict']
#         return model.load_state_dict(state_dict)

if __name__ == '__main__':
    D = DataSetHandler()
    traindf, valdf, testdf = D.initialize_classification_dataframes('c_tinyimagenet')
    m = ModelsV1()