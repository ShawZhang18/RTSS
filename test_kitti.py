import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from dataloader import KITTILoader as DA
import utils.logger as logger
import torch.backends.cudnn as cudnn

import models.anynet

import numpy as np
import cv2 as cv2

from utils.compute_seg_metric import *

os.environ['CUDA_VISIBLE_DEVICES']='0'

parser = argparse.ArgumentParser(description='Anynet fintune on KITTI')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datatype', default='2015',
                    help='datapath')
# 各种路径修改
parser.add_argument('--datapath', default='/home/zhangxiao/data/kitti2015/data_scene_flow/training/', help='datapath')
parser.add_argument('--pretrained', type=str, default='results/kitti2015_multi/models/checkpoint140.tar',
                    help='pretrained model path')
parser.add_argument('--save_path', type=str, default='results/zacao_kitti/models/',
                    help='the path of saving checkpoints and log')

parser.add_argument('--split_file', type=str, default='/home/zhangxiao/data/kitti2015/data_scene_flow/split_val.txt')

parser.add_argument('--seg_classes', type=int, default=34,
                    help='number of epochs to train')

parser.add_argument('--train_bsize', type=int, default=2,
                    help='batch size for training (default: 6)')
parser.add_argument('--test_bsize', type=int, default=2,
                    help='batch size for testing (default: 8)')

parser.add_argument('--resume', type=str, default=None,
                    help='resume path')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--with_spn', action='store_true',default=False, help='with spn network or not')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=8, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
parser.add_argument('--start_epoch_for_spn', type=int, default=121)
parser.add_argument('--is_training', type=bool)
parser.add_argument('--evaluate', action='store_true')


args = parser.parse_args()

if args.datatype == '2015':
    from dataloader import KITTIloader2015 as ls
elif args.datatype == '2012':
    from dataloader import KITTIloader2012 as ls
elif args.datatype == 'CityScape':
    from dataloader import listCityScape as ls
elif args.datatype == 'other':
    from dataloader import diy_dataset as ls


def main():
    global args
    log = logger.setup_logger(args.save_path + '/training.log')

    # train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp, test_fn = ls.dataloader(
    #     args.datapath,log, args.split_file)
    #
    # TrainImgLoader = torch.utils.data.DataLoader(
    #     DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True),
    #     batch_size=args.train_bsize, shuffle=True, num_workers=4, drop_last=False)
    #
    # TestImgLoader = torch.utils.data.DataLoader(
    #     DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False, test_fn),
    #     batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, left_val_disp, val_fn, left_train_semantic, left_val_semantic = ls.dataloader(
        args.datapath,log, args.split_file)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp, left_train_semantic, True),
        batch_size=args.train_bsize, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, left_val_disp, left_val_semantic, False, val_fn),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    model = models.anynet.AnyNet(args)
    model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            log.info("=> loaded pretrained model '{}'"
                     .format(args.pretrained))
        else:
            log.info("=> no pretrained model found at '{}'".format(args.pretrained))
            log.info("=> Will start from scratch.")
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('Not Resume')
    cudnn.benchmark = True

    test(TestImgLoader, model, log)
    return

def test(dataloader, model, log):

    stages = 3 + args.with_spn
    D1s = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.eval()

    result_all = []
    for batch_idx, (imgL, imgR, disp_L, left_val_semantic, fn) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        with torch.no_grad():
            outputs_all = model(imgL, imgR, left_val_semantic)
            output_seg = outputs_all[-1]

            outputs = outputs_all[:-1]
            for x in range(stages):
                output = torch.squeeze(outputs[x], 1)
                D1s[x].update(error_estimating(output, disp_L).item())
                output_seg = torch.squeeze(output_seg, 1)

                for i in range(output.size()[0]):
                    img_cpu = np.asarray(output.cpu())
                    # img_cpu = np.asarray(disp_L.cpu())
                    img_save = np.clip(img_cpu[i, :, :], 0, 2 ** 16)
                    img_save = (img_save).astype(np.uint16)
                    name = "/home/zhangxiao/code/AnyNet/results/zacao_kitti/png_result/" + fn[i]
                    # cv2.imshow(name, img_save)
                    cv2.imwrite(name, img_save)
                    # cv2.waitKey(0)

                    img_seg_res = np.array(output_seg.cpu())
                    img_seg_res = img_seg_res[i]
                    img_seg_save = img_seg_res.astype(np.uint8)

                    # gt = left_val_semantic[i]
                    gt = np.array(left_val_semantic.cpu()[i])
                    hist_tmp, labeled_tmp, corret_tmp = hist_info(args.seg_classes, img_seg_res, gt)
                    result_each = {'hist':hist_tmp, 'labeled':labeled_tmp, 'correct':corret_tmp}
                    result_all.append(result_each)

                    # cv2.imshow(fn[i], img_seg_save)
                    # cv2.waitKey(0)

        info_str = '\t'.join(['Stage {} = {:.4f}({:.4f})'.format(x, D1s[x].val, D1s[x].avg) for x in range(stages)])

        log.info('[{}/{}] {}'.format(
            batch_idx, length_loader, info_str))

    info_str = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(stages)])
    log.info('Average test 3-Pixel Error = ' + info_str)

    # class_names = ["back","zuowu","weed","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28"]
    class_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14","15", "16", "17", "18", "19", "20",
                   "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40"]
    result_line = compute_metric(args.seg_classes, class_names, result_all)
    print(result_line)

def error_estimating(disp, ground_truth, maxdisp=192):
    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)

    errmap = torch.abs(disp - gt)
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    return err3.float() / mask.sum().float()

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = args.lr
    elif epoch <= 400:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
