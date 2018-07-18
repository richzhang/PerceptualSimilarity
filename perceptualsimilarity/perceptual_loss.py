from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
from models import dist_model

class PerceptualLoss(torch.nn.Module):
    def __init__(self, model='net-lin', net='vgg', use_gpu=True): # VGG using our perceptually-learned weights (LPIPS metric)
    # def __init__(self, model='net', net='vgg', use_gpu=True): # "default" way of using VGG
        print('Setting up Perceptual loss..')
        self.model = dist_model.DistModel()
        self.model.initialize(model=model, net=net, use_gpu=True)
        print('Done')

    def forward(self, pred, target, normalize=False):
        """
        Pred and target are Variables.
        If normalize is on, scales images between [-1, 1]
        Assumes the inputs are in range [0, 1].
        """
        if normalize:
            target = 2 * target  - 1
            pred = 2 * pred  - 1

        dist = self.model.forward_pair(target, pred)

        return dist


if __name__ == '__main__':
    import scipy
    import numpy as np
    ref_path  = './imgs/ex_ref.png'
    pred_path = './imgs/ex_p0.png'

    ref_img = scipy.misc.imread(ref_path).transpose(2, 0, 1) / 255.
    pred_img = scipy.misc.imread(pred_path).transpose(2, 0, 1) / 255.

    # Torchify
    ref = Variable(torch.FloatTensor(ref_img).cuda(), requires_grad=False)
    pred = Variable(torch.FloatTensor(pred_img).cuda())

    # 1 x 3 x H x W
    ref = ref.unsqueeze(0)
    pred = pred.unsqueeze(0)

    loss_fn = PerceptualLoss()
    dist = loss_fn.forward(pred, ref, normalize=True)
    # print('Dist %.3f' % dist.data[0].cpu().numpy())

    # As optimization, test backprop
    class PerceptModel(torch.nn.Module):
        def __init__(self):
            super(PerceptModel, self).__init__()
            self.pred = torch.nn.Parameter(pred.data)

        def forward(self):
            return self.pred

    model = PerceptModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    import matplotlib.pyplot as plt
    plt.ion()
    fig = plt.figure(1)
    ax = fig.add_subplot(131)
    ax.imshow(ref_img.transpose(1, 2, 0))
    ax.set_title('reference')
    ax = fig.add_subplot(133)
    ax.imshow(pred_img.transpose(1, 2, 0))
    ax.set_title('orig pred')

    for i in range(1000):
        optimizer.zero_grad()
        dist = loss_fn.forward(model.forward(), ref, normalize=True)
        dist.backward()
        if i % 10 == 0:
            print('iter %d, dist %.3g' % (i, dist.view(-1).data.cpu().numpy()[0]))
            pred_img = model.pred[0].data.cpu().numpy().transpose(1, 2, 0)
            ax = fig.add_subplot(132)            
            ax.imshow(np.clip(pred_img, 0, 1))
            ax.set_title('iter %d, dist %.3g' % (i, dist.view(-1).data.cpu().numpy()[0]))
            plt.pause(5e-2)
        optimizer.step()

    # from IPython import embed; embed()
    # import ipdb; ipdb.set_trace()
