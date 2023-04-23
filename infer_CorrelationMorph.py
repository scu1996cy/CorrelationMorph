from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models.CRSTrans import CorrelationMorph

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    batch_size = 1
    test_dir = 'D:/DATA/OASIS/Test/'
    model_idx = -1
    weights = [1, 1]
    model_folder = 'CorrelationMorph_ncc_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_dir = 'experiments/' + model_folder
    img_size = (160, 192, 224)

    '''
    Initialize model
    '''
    model = CorrelationMorph()
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(img_size, 'bilinear')
    reg_model_bilin.cuda()

    '''
    Initialize training
    '''
    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    test_set = datasets.OASISBrainTestDataset(glob.glob(test_dir + '*.pkl'), transforms=val_composed)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    best = 0
    eval = utils.AverageMeter()
    with torch.no_grad():
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            x_in = torch.cat((x, y), dim=1)
            grid_img = mk_grid_img(8, 1, img_size)
            output = model(x_in)
            def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
            def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])
            #dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
            metric = utils.metric_val_VOI(def_out.long(), y_seg.long())
            eval.update(metric.item(), x.size(0))
            print(eval.avg)
    print('best{:.4f}best_std{:.4f}'.format(eval.avg,eval.std))

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    torch.manual_seed(0)
    main()