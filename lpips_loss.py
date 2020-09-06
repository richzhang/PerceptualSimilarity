
from __future__ import absolute_import

import sys
import scipy
import scipy.misc
import numpy as np
import torch
from torch.autograd import Variable
import lpips

from IPython import embed

use_gpu = False

ref_path  = './imgs/ex_ref.png'
pred_path = './imgs/ex_p1.png'

ref = lpips.im2tensor(lpips.load_image(ref_path))
pred = Variable(lpips.im2tensor(lpips.load_image(pred_path)), requires_grad=True)

loss_fn = lpips.LPIPS(net='vgg')
if(use_gpu):
    loss_fn.cuda()
optimizer = torch.optim.Adam([pred,], lr=1e-3, betas=(0.9, 0.999))

import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure(1)
ax = fig.add_subplot(131)
ax.imshow(lpips.tensor2im(ref))
ax.set_title('target')
ax = fig.add_subplot(133)
ax.imshow(lpips.tensor2im(pred.data))
ax.set_title('initialization')

for i in range(1000):
    dist = loss_fn.forward(pred, ref, normalize=False)
    optimizer.zero_grad()
    dist.backward()
    optimizer.step()
    pred.data = torch.clamp(pred.data, -1, 1)
    
    if i % 10 == 0:
        print('iter %d, dist %.3g' % (i, dist.view(-1).data.cpu().numpy()[0]))
        pred.data = torch.clamp(pred.data, -1, 1)
        pred_img = lpips.tensor2im(pred.data)
        # embed()
        ax = fig.add_subplot(132)            
        ax.imshow(pred_img)
        ax.set_title('iter %d, dist %.3f' % (i, dist.view(-1).data.cpu().numpy()[0]))
        plt.pause(5e-2)
        # plt.imsave('imgs_saved/%04d.jpg'%i,pred_img)


