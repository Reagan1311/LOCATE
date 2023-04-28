import os
import sys
import time
import shutil
import logging
import argparse

import cv2
import torch
import torch.nn as nn
import numpy as np
from models.locate import Net as model

from utils.viz import viz_pred_train, viz_pred_test
from utils.util import set_seed, process_gt, normalize_map, get_optimizer
from utils.evaluation import cal_kl, cal_sim, cal_nss, AverageMeter, compute_cls_acc

parser = argparse.ArgumentParser()
##  path
parser.add_argument('--data_root', type=str, default='/home/gen/Project/aff_grounding/dataset/AGD20K/')
parser.add_argument('--save_root', type=str, default='save_models')
parser.add_argument("--divide", type=str, default="Seen")
##  image
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--resize_size', type=int, default=256)
##  dataloader
parser.add_argument('--num_workers', type=int, default=8)
##  train
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--warm_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--show_step', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--viz', action='store_true', default=False)

#### test
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument('--test_num_workers', type=int, default=8)

args = parser.parse_args()
torch.cuda.set_device('cuda:' + args.gpu)
lr = args.lr

if args.divide == "Seen":
    aff_list = ['beat', "boxing", "brush_with", "carry", "catch", "cut", "cut_with", "drag", 'drink_with',
                "eat", "hit", "hold", "jump", "kick", "lie_on", "lift", "look_out", "open", "pack", "peel",
                "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick", "stir", "swing", "take_photo",
                "talk_on", "text_on", "throw", "type_on", "wash", "write"]
else:
    aff_list = ["carry", "catch", "cut", "cut_with", 'drink_with',
                "eat", "hit", "hold", "jump", "kick", "lie_on", "open", "peel",
                "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick",
                "swing", "take_photo", "throw", "type_on", "wash"]

if args.divide == "Seen":
    args.num_classes = 36
else:
    args.num_classes = 25

args.exocentric_root = os.path.join(args.data_root, args.divide, "trainset", "exocentric")
args.egocentric_root = os.path.join(args.data_root, args.divide, "trainset", "egocentric")
args.test_root = os.path.join(args.data_root, args.divide, "testset", "egocentric")
args.mask_root = os.path.join(args.data_root, args.divide, "testset", "GT")
time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
args.save_path = os.path.join(args.save_root, time_str)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path, exist_ok=True)
dict_args = vars(args)

shutil.copy('./models/locate.py', args.save_path)
shutil.copy('./train.py', args.save_path)

str_1 = ""
for key, value in dict_args.items():
    str_1 += key + "=" + str(value) + "\n"

logging.basicConfig(filename='%s/run.log' % args.save_path, level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.info(str_1)

if __name__ == '__main__':
    set_seed(seed=0)

    from data.datatrain import TrainData

    trainset = TrainData(exocentric_root=args.exocentric_root,
                         egocentric_root=args.egocentric_root,
                         resize_size=args.resize_size,
                         crop_size=args.crop_size, divide=args.divide)

    TrainLoader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    from data.datatest import TestData

    testset = TestData(image_root=args.test_root,
                       crop_size=args.crop_size,
                       divide=args.divide, mask_root=args.mask_root)
    TestLoader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=args.test_batch_size,
                                             shuffle=False,
                                             num_workers=args.test_num_workers,
                                             pin_memory=True)

    model = model(aff_classes=args.num_classes)
    model = model.cuda()
    model.train()
    optimizer, scheduler = get_optimizer(model, args)

    best_kld = 1000
    print('Train begining!')
    for epoch in range(args.epochs):
        model.train()
        logger.info('LR = ' + str(scheduler.get_last_lr()))
        exo_aff_acc = AverageMeter()
        ego_obj_acc = AverageMeter()

        for step, (exocentric_image, egocentric_image, aff_label) in enumerate(TrainLoader):
            aff_label = aff_label.cuda().long()  # b x n x 36
            exo = exocentric_image.cuda()  # b x n x 3 x 224 x 224
            ego = egocentric_image.cuda()

            masks, logits, loss_proto, loss_con = model(exo, ego, aff_label, (epoch, args.warm_epoch))

            exo_aff_logits = logits['aff']
            num_exo = exo.shape[1]
            exo_aff_loss = torch.zeros(1).cuda()
            for n in range(num_exo):
                exo_aff_loss += nn.CrossEntropyLoss().cuda()(exo_aff_logits[:, n], aff_label)
            exo_aff_loss /= num_exo

            loss_dict = {'ego_ce': nn.CrossEntropyLoss().cuda()(logits['aff_ego'], aff_label),
                         'exo_ce': exo_aff_loss,
                         'con_loss': loss_proto,
                         'loss_cen': loss_con * 0.07,
                         }

            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cur_batch = exo.size(0)
            exo_acc = 100. * compute_cls_acc(logits['aff'].mean(1), aff_label)
            exo_aff_acc.updata(exo_acc, cur_batch)
            metric_dict = {'exo_aff_acc': exo_aff_acc.avg}

            if (step + 1) % args.show_step == 0:
                log_str = 'epoch: %d/%d + %d/%d | ' % (epoch + 1, args.epochs, step + 1, len(TrainLoader))
                log_str += ' | '.join(['%s: %.3f' % (k, v) for k, v in metric_dict.items()])
                log_str += ' | '
                log_str += ' | '.join(['%s: %.3f' % (k, v) for k, v in loss_dict.items()])
                logger.info(log_str)

                # Visualization the prediction during training
                if args.viz:
                    viz_pred_train(args, ego, exo, masks, aff_list, aff_label, epoch, step + 1)

        scheduler.step()
        KLs = []
        SIM = []
        NSS = []
        model.eval()
        GT_path = args.divide + "_gt.t7"
        if not os.path.exists(GT_path):
            process_gt(args)
        GT_masks = torch.load(args.divide + "_gt.t7")

        for step, (image, label, mask_path) in enumerate(TestLoader):
            ego_pred = model.test_forward(image.cuda(), label.long().cuda())
            cluster_sim_maps = []
            ego_pred = np.array(ego_pred.squeeze().data.cpu())
            ego_pred = normalize_map(ego_pred, args.crop_size)

            names = mask_path[0].split("/")
            key = names[-3] + "_" + names[-2] + "_" + names[-1]
            GT_mask = GT_masks[key]
            GT_mask = GT_mask / 255.0

            GT_mask = cv2.resize(GT_mask, (args.crop_size, args.crop_size))

            kld, sim, nss = cal_kl(ego_pred, GT_mask), cal_sim(ego_pred, GT_mask), cal_nss(ego_pred, GT_mask)
            KLs.append(kld)
            SIM.append(sim)
            NSS.append(nss)

            # Visualization the prediction during evaluation
            if args.viz:
                if (step + 1) % args.show_step == 0:
                    img_name = key.split(".")[0]
                    viz_pred_test(args, image, ego_pred, GT_mask, aff_list, label, img_name, epoch)

        mKLD = sum(KLs) / len(KLs)
        mSIM = sum(SIM) / len(SIM)
        mNSS = sum(NSS) / len(NSS)

        logger.info(
            "epoch=" + str(epoch + 1) + " mKLD = " + str(round(mKLD, 3))
            + " mSIM = " + str(round(mSIM, 3)) + " mNSS = " + str(round(mNSS, 3))
            + " bestKLD = " + str(round(best_kld, 3)))

        if mKLD < best_kld:
            best_kld = mKLD
            model_name = 'best_model_' + str(epoch + 1) + '_' + str(round(best_kld, 3)) \
                         + '_' + str(round(mSIM, 3)) \
                         + '_' + str(round(mNSS, 3)) \
                         + '.pth'
            torch.save(model.state_dict(), os.path.join(args.save_path, model_name))
