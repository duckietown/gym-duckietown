from gail.train import _train, gen_expert_data
from gail.dataloader import *

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from learning.imitation.basic.train_imitation import _train
from learning.imitation.basic.enjoy_imitation import _enjoy

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

args={'episodes':3, 
      'seed':1234, 
      'steps':50, 
      'batch_size':32,
      'epochs':1000,
      'model_directory':'models/',
      'data_directory':'D:/Michael/Learning/duckietown_data2/',
      'get_samples':True}
args = Struct(**args)

# print(args.episodes)
# gen_expert_data(args)
def run():
    # torch.multiprocessing.freeze_support()
    # print('loop')
    data = ExpertTrajDataset(args)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=4)


    for i_batch, sample_batched in enumerate(dataloader):
        sample_batched_obs = sample_batched['observation'].float().to(device).reshape((args.batch_size*100, 3, 160,120))
        sample_batched_acts= sample_batched['action'].float().to(device).reshape((args.batch_size*100, 2))
        for i in sample_batched_obs:
            print(i)
            plt.imshow(i[0,:,:].to('cpu').numpy())
        print(i_batch, sample_batched_obs.size(), sample_batched_acts.size())


        if i_batch == 4:
            break

    # _trainGAIL(args)
    # _train(args)

if __name__ == '__main__':
    # gen_expert_data(args)
    # _train(args)
    # go(args)
    # # run()
    _train(args)
    _enjoy()