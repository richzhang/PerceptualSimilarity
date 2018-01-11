import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from .base_model import BaseModel
from . import networks_basic as networks
from scipy.ndimage import zoom

class DistModel(BaseModel):
    def name(self):
        return self.model_name

    def initialize(self, model='net-lin', net='squeeze', colorspace='Lab', use_gpu=True, printNet=False):
        '''
        INPUTS
            model - ['net-lin'] for linearly calibrated network
                    ['net'] for off-the-shelf network
                    ['L2'] for L2 distance in Lab colorspace
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
        '''
        BaseModel.initialize(self, use_gpu=use_gpu)

        self.model = model
        self.net = net
        self.use_gpu = use_gpu

        self.model_name = '%s [%s]'%(model,net)
        if(self.model == 'net-lin'): # pretrained net + linear layer
            self.net = networks.PNetLin(use_gpu=use_gpu,pnet_type=net,use_dropout=True)
            self.net.load_state_dict(torch.load('./weights/%s.pth'%net))
        elif(self.model=='net'): # pretrained network
            self.net = networks.PNet(use_gpu=use_gpu,pnet_type=net)
            self.is_fake_net = True
        elif(self.model in ['L2','l2']):
            self.net = networks.L2(use_gpu=use_gpu,colorspace=colorspace) # not really a network, only for testing
            self.model_name = 'L2'
        elif(self.model in ['DSSIM','dssim','SSIM','ssim']):
            self.net = networks.DSSIM(use_gpu=use_gpu,colorspace=colorspace)
            self.model_name = 'SSIM'
        else:
            raise ValueError("Model [%s] not recognized." % self.model)

        self.parameters = list(self.net.parameters())
        self.net.eval()

        if(printNet):
            print('---------- Networks initialized -------------')
            networks.print_network(self.net)
            print('-----------------------------------------------')

    def forward_pair(self,in1,in2,retPerLayer=False):
        if(retPerLayer):
            return self.net.forward(in1,in2, retPerLayer=True)
        else:
            return self.net.forward(in1,in2)

    def forward(self, in0, in1, retNumpy=True):
        ''' Function computes the distance between image patches in0 and in1
        INPUTS
            in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
            retNumpy - [False] to return as torch.Tensor, [True] to return as numpy array
        OUTPUT
            computed distances between in0 and in1
        '''

        self.input_ref = in0
        self.input_p0 = in1

        if(self.use_gpu):
            self.input_ref = self.input_ref.cuda()
            self.input_p0 = self.input_p0.cuda()

        self.var_ref = Variable(self.input_ref,requires_grad=True)
        self.var_p0 = Variable(self.input_p0,requires_grad=True)

        self.d0 = self.forward_pair(self.var_ref, self.var_p0)
        self.loss_total = self.d0

        if(retNumpy):
            return self.d0.cpu().data.numpy().flatten()
        else:
            return self.d0


def score_2afc_dataset(data_loader,func):
    ''' Function computes Two Alternative Forced Choice (2AFC) score using
        distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a TwoAFCDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return numpy array of length N
    OUTPUTS
        [0] - 2AFC score in [0,1], fraction of time func agrees with human evaluators
        [1] - dictionary with following elements
            d0s,d1s - N arrays containing distances between reference patch to perturbed patches 
            gts - N array in [0,1], preferred patch selected by human evaluators
                (closer to "0" for left patch p0, "1" for right patch p1,
                "0.6" means 60pct people preferred right patch, 40pct preferred left)
            scores - N array in [0,1], corresponding to what percentage function agreed with humans
    CONSTS
        N - number of test triplets in data_loader
    '''

    d0s = []
    d1s = []
    gts = []

    # bar = pb.ProgressBar(max_value=data_loader.load_data().__len__())
    for (i,data) in enumerate(data_loader.load_data()):
        d0s+=func(data['ref'],data['p0']).tolist()
        d1s+=func(data['ref'],data['p1']).tolist()
        gts+=data['judge'].cpu().numpy().flatten().tolist()
        # bar.update(i)

    d0s = np.array(d0s)
    d1s = np.array(d1s)
    gts = np.array(gts)
    scores = (d0s<d1s)*(1.-gts) + (d1s<d0s)*gts + (d1s==d0s)*.5

    return(np.mean(scores), dict(d0s=d0s,d1s=d1s,gts=gts,scores=scores))

def score_jnd_dataset(data_loader,func):
    ''' Function computes JND score using distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a JNDDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return numpy array of length N
    OUTPUTS
        [0] - JND score in [0,1], mAP score (area under precision-recall curve)
        [1] - dictionary with following elements
            ds - N array containing distances between two patches shown to human evaluator
            sames - N array containing fraction of people who thought the two patches were identical
    CONSTS
        N - number of test triplets in data_loader
    '''

    ds = []
    gts = []

    # bar = pb.ProgressBar(max_value=data_loader.load_data().__len__())
    for (i,data) in enumerate(data_loader.load_data()):
        ds+=func(data['p0'],data['p1']).tolist()
        gts+=data['same'].cpu().numpy().flatten().tolist()
        # bar.update(i)

    sames = np.array(gts)
    ds = np.array(ds)

    sorted_inds = np.argsort(ds)
    ds_sorted = ds[sorted_inds]
    sames_sorted = sames[sorted_inds]

    TPs = np.cumsum(sames_sorted)
    FPs = np.cumsum(1-sames_sorted)
    FNs = np.sum(sames_sorted)-TPs

    precs = TPs/(TPs+FPs)
    recs = TPs/(TPs+FNs)
    score = util.voc_ap(recs,precs)

    return(score, dict(ds=ds,sames=sames))
