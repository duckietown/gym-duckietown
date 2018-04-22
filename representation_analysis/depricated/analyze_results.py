import torch
from matplotlib import pyplot as plt
import numpy as np

# Save the checkpoint
loaded_state = torch.load('representation_analysis/.saves/beta-vae/beta-vae_3.ckpt')
epoch = loaded_state['epoch']
model = loaded_state['model']
optimizer = loaded_state['optimizer']
total_losses = loaded_state['loss']['total']
reconstruction_losses = loaded_state['loss']['reconstruction']
kl_divergence_losses = loaded_state['loss']['kl_divergence']
args = loaded_state['args']
beta = loaded_state['beta']

plt.figure()
plt.title('Losses Over Time')
plt.plot(np.array(reconstruction_losses[3:]), color='red', label='Reconstruction Loss')
plt.plot(np.array(total_losses[3:]), color='blue', label='Total Loss')
plt.plot(np.array(kl_divergence_losses[3:]), color='green', label='KL Divergence Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend(loc='lower left')
plt.savefig('representation_analysis/results/losses')
