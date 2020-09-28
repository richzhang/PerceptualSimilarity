import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import argparse
import lpips

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ref_path', type=str, default='./imgs/ex_ref.png')
parser.add_argument('--pred_path', type=str, default='./imgs/ex_p1.png')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

loss_fn = lpips.LPIPS(net='vgg')
if(opt.use_gpu):
    loss_fn.cuda()

ref = lpips.im2tensor(lpips.load_image(opt.ref_path))
pred = Variable(lpips.im2tensor(lpips.load_image(opt.pred_path)), requires_grad=True)
if(opt.use_gpu):
    with torch.no_grad():
        ref = ref.cuda()
        pred = pred.cuda()

optimizer = torch.optim.Adam([pred,], lr=1e-3, betas=(0.9, 0.999))

plt.ion()
fig = plt.figure(1)
ax = fig.add_subplot(131)
ax.imshow(lpips.tensor2im(ref))
ax.set_title('target')
ax = fig.add_subplot(133)
ax.imshow(lpips.tensor2im(pred.data))
ax.set_title('initialization')

for i in range(1000):
    dist = loss_fn.forward(pred, ref)
    optimizer.zero_grad()
    dist.backward()
    optimizer.step()
    pred.data = torch.clamp(pred.data, -1, 1)
    
    if i % 10 == 0:
        print('iter %d, dist %.3g' % (i, dist.view(-1).data.cpu().numpy()[0]))
        pred.data = torch.clamp(pred.data, -1, 1)
        pred_img = lpips.tensor2im(pred.data)

        ax = fig.add_subplot(132)            
        ax.imshow(pred_img)
        ax.set_title('iter %d, dist %.3f' % (i, dist.view(-1).data.cpu().numpy()[0]))
        plt.pause(5e-2)
        # plt.imsave('imgs_saved/%04d.jpg'%i,pred_img)


