import os
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy

import torch
import torchvision
from torch import nn
import open_clip

from torchvision.datasets import CIFAR100


def create(image_list, data_list, save_folder):
    assert image_list is not None, "image_list must be provided to generate features"
    embed_size=512
    seg_maps = []
    total_lengths = []
    timer = 0
    img_embeds = torch.zeros((len(image_list), 300, embed_size))
    seg_maps = torch.zeros((len(image_list), 4, *image_list[0].shape[1:])) 
    mask_generator.predictor.model.to('cuda')

    for i, img in tqdm(enumerate(image_list), desc="Embedding images", leave=False):
        timer += 1
        try:
            img_embed, seg_map = _embed_clip_sam_tiles(img.unsqueeze(0), sam_encoder)
        except:
            raise ValueError(timer)

        lengths = [len(v) for k, v in img_embed.items()]
        total_length = sum(lengths)
        total_lengths.append(total_length)
        
        if total_length > img_embeds.shape[1]:
            pad = total_length - img_embeds.shape[1]
            img_embeds = torch.cat([
                img_embeds,
                torch.zeros((len(image_list), pad, embed_size))
            ], dim=1)

        img_embed = torch.cat([v for k, v in img_embed.items()], dim=0)
        assert img_embed.shape[0] == total_length
        img_embeds[i, :total_length] = img_embed
        
        seg_map_tensor = []
        lengths_cumsum = lengths.copy()
        for j in range(1, len(lengths)):
            lengths_cumsum[j] += lengths_cumsum[j-1]
        for j, (k, v) in enumerate(seg_map.items()):
            if j == 0:
                seg_map_tensor.append(torch.from_numpy(v))
                continue
            assert v.max() == lengths[j] - 1, f"{j}, {v.max()}, {lengths[j]-1}"
            v[v != -1] += lengths_cumsum[j-1]
            seg_map_tensor.append(torch.from_numpy(v))
        seg_map = torch.stack(seg_map_tensor, dim=0)
        seg_maps[i] = seg_map

    mask_generator.predictor.model.to('cpu')
        
    for i in range(img_embeds.shape[0]):
        save_path = os.path.join(save_folder, data_list[i].split('.')[0])
        assert total_lengths[i] == int(seg_maps[i].max() + 1)
        curr = {
            'feature': img_embeds[i, :total_lengths[i]],
            'seg_maps': seg_maps[i]
        }
        sava_numpy(save_path, curr)

def sava_numpy(save_path, data):
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, data['seg_maps'].numpy())
    np.save(save_path_f, data['feature'].numpy())

def _embed_clip_sam_tiles(image, sam_encoder):
    aug_imgs = torch.cat([image])
    seg_images, seg_map = sam_encoder(aug_imgs)

    clip_embeds = {}

    tiles = seg_images['default']
    tiles = tiles.to("cuda")
    with torch.no_grad():
        clip_embed = model.encode_image(tiles)
    clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
    clip_embeds['default'] = clip_embed.detach().cpu().half()
    
    return clip_embeds, seg_map

def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def sam_encoder(image):
    image = cv2.cvtColor(image[0].permute(1,2,0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    # pre-compute masks
    #masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
    masks = mask_generator.generate(image)

    # pre-compute postprocess
    masks = masks_update(masks, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    
    def mask2segmap(masks, image):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks[0])):
            mask = masks[0][i]
            seg_img = get_seg_img(mask, image)
            pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))

            predict(pad_seg_img.transpose(2,0,1))

            seg_img_list.append(pad_seg_img)

            seg_map[masks[0][i]['segmentation']] = i
        seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')

        return seg_imgs, seg_map

    seg_images, seg_maps = {}, {}
    seg_images['default'], seg_maps['default'] = mask2segmap(masks, image)

    # 0:default 1:s 2:m 3:l
    return seg_images, seg_maps

def predict(seg_img):
    image_input = torch.tensor(np.stack(seg_img)).cuda().float()
    image_input = image_input.unsqueeze(0)

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)
    return top_probs, top_labels

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="bed")
    parser.add_argument('--adv_object', type=str, default="banana")
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_name', type=str, default="sam_vit_h_4b8939.pth")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = os.path.join("data/" + args.dataset_name)
    sam_ckpt_path = os.path.join("sam_ckpt/" + args.sam_ckpt_name)
    img_folder = os.path.join(dataset_path, 'images')
    data_list = os.listdir(img_folder)
    data_list.sort()

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device="cuda")
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    cifar100 = CIFAR100(os.path.expanduser("~/.cache"), transform=preprocess, download=True)
    adv_object_index = cifar100.classes.index(args.adv_object) if args.adv_object in cifar100.classes else (cifar100.classes.append(args.adv_object), 100)[1]
    text_descriptions = [f"This is a photo of a {label}" for label in cifar100.classes]
    text_tokens = tokenizer(text_descriptions).cuda()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float().cuda()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    img_list = []

    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)

        orig_w, orig_h = image.shape[1], image.shape[0]
        if args.resolution == -1:
            if orig_h > 1080:
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
            
        scale = float(global_down)
        resolution = (int( orig_w  / scale), int(orig_h / scale))
        
        image = cv2.resize(image, resolution)
        image = torch.from_numpy(image)
        img_list.append(image)
    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    imgs = torch.cat(images)

    save_folder = os.path.join(dataset_path, 'noisy_images')
    os.makedirs(save_folder, exist_ok=True)
    create(imgs, data_list, save_folder)