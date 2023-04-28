import os
import torch
import numpy as np
from PIL import Image
from utils.util import normalize_map, overlay_mask
import matplotlib.pyplot as plt


# visualize the prediction of the first batch
def viz_pred_train(args, ego, exo, masks, aff_list, aff_label, epoch, step):
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=ego.dtype, device=ego.device).view(-1, 1, 1)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=ego.dtype, device=ego.device).view(-1, 1, 1)

    ego_0 = ego[0].squeeze(0) * std + mean
    ego_0 = ego_0.detach().cpu().numpy() * 255
    ego_0 = Image.fromarray(ego_0.transpose(1, 2, 0).astype(np.uint8))

    exo_img = []
    num_exo = exo.shape[1]
    for i in range(num_exo):
        name = 'exo_' + str(i)
        locals()[name] = exo[0][i].squeeze(0) * std + mean
        locals()[name] = locals()[name].detach().cpu().numpy() * 255
        locals()[name] = Image.fromarray(locals()[name].transpose(1, 2, 0).astype(np.uint8))
        exo_img.append(locals()[name])

    exo_cam = masks['exo_aff'][0]

    sim_maps, exo_sim_maps, part_score, ego_pred = masks['pred']
    num_clu = sim_maps.shape[1]
    part_score = np.array(part_score[0].squeeze().data.cpu())

    ego_pred = np.array(ego_pred[0].squeeze().data.cpu())
    ego_pred = normalize_map(ego_pred, args.crop_size)
    ego_pred = Image.fromarray(ego_pred)
    ego_pred = overlay_mask(ego_0, ego_pred, alpha=0.5)

    ego_sam = masks['ego_sam']
    ego_sam = np.array(ego_sam[0].squeeze().data.cpu())
    ego_sam = normalize_map(ego_sam, args.crop_size)
    ego_sam = Image.fromarray(ego_sam)
    ego_sam = overlay_mask(ego_0, ego_sam, alpha=0.1)

    aff_str = aff_list[aff_label[0].item()]

    for i in range(num_exo):
        name = 'exo_aff' + str(i)
        locals()[name] = np.array(exo_cam[i].squeeze().data.cpu())
        locals()[name] = normalize_map(locals()[name], args.crop_size)
        locals()[name] = Image.fromarray(locals()[name])
        locals()[name] = overlay_mask(exo_img[i], locals()[name], alpha=0.5)

    for i in range(num_clu):
        name = 'sim_map' + str(i)
        locals()[name] = np.array(sim_maps[0][i].squeeze().data.cpu())
        locals()[name] = normalize_map(locals()[name], args.crop_size)
        locals()[name] = Image.fromarray(locals()[name])
        locals()[name] = overlay_mask(ego_0, locals()[name], alpha=0.5)

        # Similarity maps for the first exocentric image
        name = 'exo_sim_map' + str(i)
        locals()[name] = np.array(exo_sim_maps[0, 0][i].squeeze().data.cpu())
        locals()[name] = normalize_map(locals()[name], args.crop_size)
        locals()[name] = Image.fromarray(locals()[name])
        locals()[name] = overlay_mask(locals()['exo_' + str(0)], locals()[name], alpha=0.5)

    # Exo&Ego plots
    fig, ax = plt.subplots(4, max(num_clu, num_exo), figsize=(8, 8))
    for axi in ax.ravel():
        axi.set_axis_off()
    for k in range(num_exo):
        ax[0, k].imshow(eval('exo_aff' + str(k)))
        ax[0, k].set_title("exo_" + aff_str)
    for k in range(num_clu):
        ax[1, k].imshow(eval('sim_map' + str(k)))
        ax[1, k].set_title('PartIoU_' + str(round(part_score[k], 2)))
        ax[2, k].imshow(eval('exo_sim_map' + str(k)))
        ax[2, k].set_title('sim_map_' + str(k))
    ax[3, 0].imshow(ego_pred)
    ax[3, 0].set_title(aff_str)
    ax[3, 1].imshow(ego_sam)
    ax[3, 1].set_title('Saliency')

    os.makedirs(os.path.join(args.save_path, 'viz_train'), exist_ok=True)
    fig_name = os.path.join(args.save_path, 'viz_train', 'cam_' + str(epoch) + '_' + str(step) + '.jpg')
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()


def viz_pred_test(args, image, ego_pred, GT_mask, aff_list, aff_label, img_name, epoch=None):
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img = image.squeeze(0) * std + mean
    img = img.detach().cpu().numpy() * 255
    img = Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8))

    gt = Image.fromarray(GT_mask)
    gt_result = overlay_mask(img, gt, alpha=0.5)
    aff_str = aff_list[aff_label.item()]

    ego_pred = Image.fromarray(ego_pred)
    ego_pred = overlay_mask(img, ego_pred, alpha=0.5)

    fig, ax = plt.subplots(1, 3, figsize=(10, 6))
    for axi in ax.ravel():
        axi.set_axis_off()
    ax[0].imshow(img)
    ax[0].set_title('ego')
    ax[1].imshow(ego_pred)
    ax[1].set_title(aff_str)
    ax[2].imshow(gt_result)
    ax[2].set_title('GT')

    os.makedirs(os.path.join(args.save_path, 'viz_test'), exist_ok=True)
    if epoch:
        fig_name = os.path.join(args.save_path, 'viz_test', "epoch" + str(epoch) + '_' + img_name + '.jpg')
    else:
        fig_name = os.path.join(args.save_path, 'viz_test', img_name + '.jpg')
    plt.savefig(fig_name)
    plt.close()
