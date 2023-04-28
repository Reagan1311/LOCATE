import os
import argparse
from tqdm import tqdm

import cv2
import torch
import numpy as np
from models.locate import Net as model

from utils.viz import viz_pred_test
from utils.util import set_seed, process_gt, normalize_map
from utils.evaluation import cal_kl, cal_sim, cal_nss

parser = argparse.ArgumentParser()
##  path
parser.add_argument('--data_root', type=str, default='/home/gen/Project/aff_grounding/dataset/AGD20K/')
parser.add_argument('--model_file', type=str, default=None)
parser.add_argument('--save_path', type=str, default='./save_preds')
parser.add_argument("--divide", type=str, default="Seen")
##  image
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--resize_size', type=int, default=256)
#### test
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument('--test_num_workers', type=int, default=8)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--viz', action='store_true', default=False)

args = parser.parse_args()

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

args.test_root = os.path.join(args.data_root, args.divide, "testset", "egocentric")
args.mask_root = os.path.join(args.data_root, args.divide, "testset", "GT")

if args.viz:
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

if __name__ == '__main__':
    set_seed(seed=0)

    from data.datatest import TestData

    testset = TestData(image_root=args.test_root,
                       crop_size=args.crop_size,
                       divide=args.divide, mask_root=args.mask_root)
    TestLoader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=args.test_batch_size,
                                             shuffle=False,
                                             num_workers=args.test_num_workers,
                                             pin_memory=True)

    model = model(aff_classes=args.num_classes).cuda()

    KLs = []
    SIM = []
    NSS = []
    model.eval()
    assert os.path.exists(args.model_file), "Please provide the correct model file for testing"
    model.load_state_dict(torch.load(args.model_file))

    GT_path = args.divide + "_gt.t7"
    if not os.path.exists(GT_path):
        process_gt(args)
    GT_masks = torch.load(args.divide + "_gt.t7")

    for step, (image, label, mask_path) in enumerate(tqdm(TestLoader)):
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

        if args.viz:
            img_name = key.split(".")[0]
            viz_pred_test(args, image, ego_pred, GT_mask, aff_list, label, img_name)

    mKLD = sum(KLs) / len(KLs)
    mSIM = sum(SIM) / len(SIM)
    mNSS = sum(NSS) / len(NSS)

    print(f"KLD = {round(mKLD, 3)}\nSIM = {round(mSIM, 3)}\nNSS = {round(mNSS, 3)}")
