
from __future__ import absolute_import

import sys
import scipy
import scipy.misc
import numpy as np
import torch
from torch.autograd import Variable
import models

use_gpu = True

ref_path  = './imgs/ex_ref.png'
pred_path = './imgs/ex_p1.png'

ref_img = scipy.misc.imread(ref_path).transpose(2, 0, 1) / 255.
pred_img = scipy.misc.imread(pred_path).transpose(2, 0, 1) / 255.

# Torchify
ref = Variable(torch.FloatTensor(ref_img)[None,:,:,:])
pred = Variable(torch.FloatTensor(pred_img)[None,:,:,:], requires_grad=True)

loss_fn = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=use_gpu)
optimizer = torch.optim.Adam([pred,], lr=1e-3, betas=(0.9, 0.999))

import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure(1)
ax = fig.add_subplot(131)
ax.imshow(ref_img.transpose(1, 2, 0))
ax.set_title('target')
ax = fig.add_subplot(133)
ax.imshow(pred_img.transpose(1, 2, 0))
ax.set_title('initialization')

for i in range(1000):
    dist = loss_fn.forward(pred, ref, normalize=True)
    optimizer.zero_grad()
    dist.backward()
    optimizer.step()
    pred.data = torch.clamp(pred.data, 0, 1)
    
    if i % 10 == 0:
        print('iter %d, dist %.3g' % (i, dist.view(-1).data.cpu().numpy()[0]))
        pred_img = pred[0].data.cpu().numpy().transpose(1, 2, 0)
        pred_img = np.clip(pred_img, 0, 1)
        ax = fig.add_subplot(132)            
        ax.imshow(pred_img)
        ax.set_title('iter %d, dist %.3f' % (i, dist.view(-1).data.cpu().numpy()[0]))
        plt.pause(5e-2)
        # plt.imsave('imgs_saved/%04d.jpg'%i,pred_img)


