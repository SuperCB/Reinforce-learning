'''
Function:
    Define the deep q network
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import torch
import torch.nn as nn


'''define the network'''
class DeepQNetwork(nn.Module):
    def __init__(self, imagesize, in_channels=4, num_actions=2, **kwargs):
        super(DeepQNetwork, self).__init__()
        assert imagesize == (80, 80), 'imagesize should be (80, 80), or redesign the deepQNetwork please'
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.fcs = nn.Sequential(
            nn.Linear(in_features=6400, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=num_actions)
        )
        
        
    '''forward'''
    def forward(self, x):
        x = self.convs(x)
        x = self.fcs(x.reshape(x.size(0), -1))
        return x