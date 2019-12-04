import os
import torch
from torch.autograd import Variable
from pdb import set_trace as st
from IPython import embed

class BaseModel():
    input_ = None
    save_dir = None

    def __init__(self):
        pass

    def name(self):
        return 'BaseModel'

    def initialize(self, use_gpu=True, gpu_ids=[0]):
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids

    def forward(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input_

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, path, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    def update_learning_rate(self):
        pass
