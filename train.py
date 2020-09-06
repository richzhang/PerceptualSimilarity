import torch.backends.cudnn as cudnn
cudnn.benchmark=False

import numpy as np
import time
import os
import lpips
from data import data_loader as dl
import argparse
from util.visualizer import Visualizer
from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, nargs='+', default=['train/traditional','train/cnn','train/mix'], help='datasets to train on: [train/traditional],[train/cnn],[train/mix],[val/traditional],[val/cnn],[val/color],[val/deblur],[val/frameinterp],[val/superres]')
parser.add_argument('--model', type=str, default='lpips', help='distance model type [lpips] for linearly calibrated net, [baseline] for off-the-shelf network, [l2] for euclidean distance, [ssim] for Structured Similarity Image Metric')
parser.add_argument('--net', type=str, default='alex', help='[squeeze], [alex], or [vgg] for network architectures')
parser.add_argument('--batch_size', type=int, default=50, help='batch size to test image patches in')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='gpus to use')

parser.add_argument('--nThreads', type=int, default=4, help='number of threads to use in data loader')
parser.add_argument('--nepoch', type=int, default=5, help='# epochs at base learning rate')
parser.add_argument('--nepoch_decay', type=int, default=5, help='# additional epochs at linearly learning rate')
parser.add_argument('--display_freq', type=int, default=5000, help='frequency (in instances) of showing training results on screen')
parser.add_argument('--print_freq', type=int, default=5000, help='frequency (in instances) of showing training results on console')
parser.add_argument('--save_latest_freq', type=int, default=20000, help='frequency (in instances) of saving the latest results')
parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--display_id', type=int, default=0, help='window id of the visdom display, [0] for no displaying')
parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
parser.add_argument('--display_port', type=int, default=8001,  help='visdom display port')
parser.add_argument('--use_html', action='store_true', help='save off html pages')
parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='checkpoints directory')
parser.add_argument('--name', type=str, default='tmp', help='directory name for training')

parser.add_argument('--from_scratch', action='store_true', help='model was initialized from scratch')
parser.add_argument('--train_trunk', action='store_true', help='model trunk was trained/tuned')
parser.add_argument('--train_plot', action='store_true', help='plot saving')

opt = parser.parse_args()
opt.save_dir = os.path.join(opt.checkpoints_dir,opt.name)
if(not os.path.exists(opt.save_dir)):
    os.mkdir(opt.save_dir)

# initialize model
trainer = lpips.Trainer()
trainer.initialize(model=opt.model, net=opt.net, use_gpu=opt.use_gpu, is_train=True, 
    pnet_rand=opt.from_scratch, pnet_tune=opt.train_trunk, gpu_ids=opt.gpu_ids)

# load data from all training sets
data_loader = dl.CreateDataLoader(opt.datasets,dataset_mode='2afc', batch_size=opt.batch_size, serial_batches=False, nThreads=opt.nThreads)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
D = len(dataset)
print('Loading %i instances from'%dataset_size,opt.datasets)
visualizer = Visualizer(opt)

total_steps = 0
fid = open(os.path.join(opt.checkpoints_dir,opt.name,'train_log.txt'),'w+')
for epoch in range(1, opt.nepoch + opt.nepoch_decay + 1):
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batch_size
        epoch_iter = total_steps - dataset_size * (epoch - 1)

        trainer.set_input(data)
        trainer.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(trainer.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = trainer.get_current_errors()
            t = (time.time()-iter_start_time)/opt.batch_size
            t2o = (time.time()-epoch_start_time)/3600.
            t2 = t2o*D/(i+.0001)
            visualizer.print_current_errors(epoch, epoch_iter, errors, t, t2=t2, t2o=t2o, fid=fid)

            for key in errors.keys():
                visualizer.plot_current_errors_save(epoch, float(epoch_iter)/dataset_size, opt, errors, keys=[key,], name=key, to_plot=opt.train_plot)

            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            trainer.save(opt.save_dir, 'latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        trainer.save(opt.save_dir, 'latest')
        trainer.save(opt.save_dir, epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.nepoch + opt.nepoch_decay, time.time() - epoch_start_time))

    if epoch > opt.nepoch:
        trainer.update_learning_rate(opt.nepoch_decay)

# trainer.save_done(True)
fid.close()
