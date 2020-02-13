import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

import torch.nn.init as init
import numpy as np

class Squeezenet(nn.Module):
    """
    A class used to define action regressor model based on squeezenet arch.
    ...
    Methods
    -------
    forward(images)
        makes a model forward pass on input images
    
    loss(*args)
        takes images and target action to compute the loss function used in optimization

    predict(*args)
        takes images as input and predict the action space unnormalized
    """

    def __init__(self, num_outputs=2, max_velocity = 0.7, max_steering=np.pi/2):
        """
        Parameters
        ----------
        num_outputs : int
            number of outputs of the action space (default 2)
        max_velocity : float
            the maximum velocity used by the teacher (default 0.7)
        max_steering : float
            maximum steering angle as we are predicting normalized [0-1] (default pi/2)
        """
        super(Squeezenet, self).__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.squeezenet1_1()
        self.num_outputs = num_outputs
        self.max_velocity_tensor = torch.tensor(max_velocity).to(self._device)
        self.max_steering = max_steering

        # using a subset of full squeezenet for input image features
        self.model.features = nn.Sequential(*list(self.model.features.children())[:6])
        self.final_conv = nn.Conv2d(32, self.num_outputs, kernel_size=1, stride=1)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.15),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.Dropout(p=0.15),
            self.final_conv,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.model.num_classes = self.num_outputs
        self._init_weights()

    def _init_weights(self):
        for m in self.model.classifier.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, images):
        """
        Parameters
        ----------
        images : tensor
            batch of input images to get normalized predicted action
        Returns
        -------
        action: tensor
            normalized predicted action from the model
        """
        action = self.model(images)
        return action

    def loss(self, *args):
        """
        Parameters
        ----------
        *args : 
            takes batch of images and target action space to get the loss function.
        Returns
        -------
        loss: tensor
            loss function used by the optimizer to update the model
        """
        self.train()
        images, target = args
        prediction = self.forward(images) 
        if self.num_outputs==1:
            loss = F.mse_loss(prediction,target[:,-1].reshape(-1,1), reduction='mean')
        else:
            loss_omega = F.mse_loss(prediction[:,1], target[:,1], reduction='mean')
            loss_v = F.mse_loss(prediction[:,0], target[:,0], reduction='mean')
            loss = 0.2 * loss_v + 0.8 * loss_omega
        return loss
    
    def predict(self, *args):
        """
        Parameters
        ----------
        *args : tensor
            batch of input images to get unnormalized predicted action
        Returns
        -------
        action: tensor
            action having velocity and omega of shape (batch_size, 2)
        """
        images = args[0]
        output = self.model(images)
        if self.num_outputs==1:
            omega = output
            v_tensor = self.max_velocity_tensor.clone()
        else:
            v_tensor = output[:,0].unsqueeze(1)
            omega = output[:,1].unsqueeze(1) * self.max_steering
        output = torch.cat((v_tensor, omega), 1).squeeze().detach()
        return output

if __name__ == '__main__':
    #TODO test the model input and output
    batch_size = 2
    img_size = (100, 80)
    model = Squeezenet()
    input_image = torch.rand((batch_size,3,img_size[0], img_size[1])).to(model._device)
    prediction = model.predict(input_image)
    assert list(prediction.shape) == [batch_size,model.num_outputs]