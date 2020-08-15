import datetime

import numpy as np
from tqdm import tqdm
import cv2
import os

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.tensorboard import SummaryWriter

from PIL import Image

from learning.imitation.iil_dagger.model import Squeezenet
from learning.imitation.iil_dagger.utils import MemoryMapDataset
from learning.imitation.iil_dagger.utils.save_manager import SaveManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuralNetworkPolicy:
    """
    A wrapper to train neural network model
    ...
    Methods
    -------
    optimize(observations, expert_actions, episode)
        train the model on the newly collected data from the simulator
    
    predict(observation)
        takes images and predicts the action space using the model

    save
        save a model checkpoint to storage location
    """

    def __init__(self, max_velocity, input_shape, save_path, save_manager=None, gain=10, learning_rate=1e-3, **kwargs):
        """
        Parameters
        ----------
        model : 
            input pytorch model that will be trained
        optimizer : 
            torch optimizer
        storage_location : string
            path of the model to be saved , the dataset and tensorboard logs
        dataset :
            object storing observation and labels from expert
        """
        self._train_iteration = 0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gain = gain

        # Optional parameters TODO remove this
        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch_size', 32)
        self.max_velocity = kwargs.get('max_velocity', 0.7)
        self.input_shape = input_shape

        # Base parameters
        self.model = Squeezenet(num_outputs=2, max_velocity=max_velocity).to(self._device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.save_manager = save_manager

        if self.save_manager is not None:
            self.writer = SummaryWriter(str(self.save_manager))

        self.episode = 0

        self.dataset = MemoryMapDataset(25000, (3, *input_shape), (2,), save_path)
        # Load previous weights
        if 'model_path' in kwargs:
            self.model.load_state_dict(torch.load(kwargs.get('model_path'), map_location=self._device))
            print('Loaded ')

    def __del__(self):
        self.writer.close()

    def optimize(self, observations, expert_actions, episode):
        """
        Parameters
        ----------
        observations : 
            input images collected from the simulator
        expert_actions : 
            list of actions each action [velocity, steering_angle] from expert
        episode : int
            current episode number
        """
        print('Starting episode #', str(episode))
        self.episode = episode
        self.model.episode = episode
        # Transform newly received data
        observations, expert_actions = self._transform(observations, expert_actions)

        # Retrieve data loader
        dataloader = self._get_dataloader(observations, expert_actions)

        # Train model
        for epoch in tqdm(range(1, self.epochs + 1)):
            running_loss = 0.0
            self.model.epoch = 0
            for i, data in enumerate(dataloader, 0):
                # Send data to device
                data = [variable.to(self._device) for variable in data]

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                loss = self.model.loss(*data)
                loss.backward()
                self.optimizer.step()

                # Statistics
                running_loss += loss.item()

            # Logging
            self._logging(loss=running_loss, index=epoch)

        loss_divider = 32   # magic number; can be changed; just to avoid the loss increasing over wrt number of samples
        if len(self.dataset) < loss_divider:
            q = 1
        else:
            q, r = divmod(len(self.dataset), loss_divider)
            q += 1 if r != 0 else 0
        self._logging(is_epoch=False, loss=running_loss / q, index=episode)

        # Post training routine
        self._on_optimization_end()

    def mass_predict(self, observations):   # FIXME with torch no grad
        """
        Parameters
        ----------
        observations :
            input images; already transformed
        Returns
        -------
        predictions
        """
        pred = self.model(observations)
        return pred.detach()

    def predict(self, observation, already_transformed=False):
        """
        Parameters
        ----------
        already_transformed :
            is the image already a tensor on the device
        observations :
            input image from the simulator
        Returns
        ----------
        prediction:
            action space of input observation
        """
        if not already_transformed:
            # Apply transformations to data
            observation, _ = self._transform([observation], [0])
            observation = torch.tensor(observation).to(self._device)
        # Predict with model
        prediction = self.model.predict(observation)
        prediction = prediction.detach().cpu().numpy()

        return prediction

    def save(self, filename="model.pt"):
        try:
            torch.save(self.model.state_dict(), self.save_manager.get_filepath(filename))
        except:
            os.mkdir(os.getcwd() + "/" + str(self.save_manager))
            torch.save(self.model.state_dict(), self.save_manager.get_filepath(filename))

    def _transform(self, observations, expert_actions):
        # Resize images
        observations = [Image.fromarray(cv2.resize(observation, dsize=self.input_shape[::-1])) for observation in
                        observations]
        # Transform to tensors
        compose_obs = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # using imagenet normalization values
        ])

        observations = [compose_obs(observation).numpy() for observation in observations]
        try:
            # scaling velocity to become in 0-1 range which is multiplied by max speed to get actual vel
            # also scaling steering angle to become in range -1 to 1 to make it easier to regress
            expert_actions = [np.array([expert_action[0], expert_action[1] / (np.pi / 2)]) for expert_action in
                              expert_actions]
        except:
            pass
        expert_actions = [torch.tensor(expert_action).numpy() for expert_action in expert_actions]

        return observations, expert_actions

    def _get_dataloader(self, observations, expert_actions):
        # Include new experiences
        self.dataset.extend(observations, expert_actions)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def _logging(self, is_epoch=True, index=0, loss=0.0):
        if is_epoch:
            self.writer.add_scalar('Loss/train/epoch', loss, index)
            self.writer.add_scalar('Loss/train/epoch/{}'.format(self._train_iteration), loss, index)
        else:
            self.writer.add_scalar("Loss/train/episode", loss, index)

    def _on_optimization_end(self):
        self._train_iteration += 1
