from gail.train import _train
from gail.dataloader import *
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from learning.imitation.basic.train_imitation import _train
from learning.imitation.basic.enjoy_imitation import _enjoy

from torchvision.models import resnet50
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

args={'episodes':9, 
      'seed':1234, 
      'steps':50, 
      'batch_size':15,
      'epochs':10,
      'model_directory':'models/',
      'data_directory':'D:/Michael/Learning/duckietown_data/',
      'get_samples':True,
      'train':True,
      'lrG': 0.0004,
      'lrD':0.0004}
args = Struct(**args)

if __name__ == '__main__':
    if args.train:
        _train(args)
    _enjoy()
