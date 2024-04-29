import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import clip
import json
import urllib
from torchvision.transforms.functional import to_pil_image
from PIL import Image

def predict(image):
    with torch.no_grad():
        image = transform(to_pil_image(image)).unsqueeze(0).to("cuda:1")

        outputs, _ = model(image, text)
        probs = outputs.softmax(dim=-1)

        res = probs[0,true_label]
        top5_probs, top5_inds = probs.topk(5, dim=-1)
        top5_probs = top5_probs.detach().cpu().numpy()
        top5_inds = top5_inds.detach().cpu().numpy()

        print(f"True label : {imagenet_classes[true_label]}, {res:.5f} Pred 1: {imagenet_classes[top5_inds[0, 0]]}, {top5_probs[0, 0]:.5f} 2: {imagenet_classes[top5_inds[0, 1]]}, {top5_probs[0, 1]:.5f} 3: {imagenet_classes[top5_inds[0, 2]]}, {top5_probs[0, 2]:.5f} 4: {imagenet_classes[top5_inds[0, 3]]}, {top5_probs[0, 3]:.5f} 5: {imagenet_classes[top5_inds[0, 4]]}, {top5_probs[0, 4]:.5f}")

    return res, top5_inds

def find_or_add_label(label, class_list):
    if label in class_list:
        return class_list.index(label)
    else:
        class_list.append(label)
        return len(class_list) - 1

def unnormalize(im):
    tensor = torch.tensor(im).float().cpu()
    tensor = tensor.clone().detach().unsqueeze(0)
    tensor.mul_(std).add_(mean)
    tensor = tensor.cpu().squeeze().permute(1, 2, 0)
    return tensor


def run(image_list_plain, image_list_noisy):
 
    result_gt = 0.0
    result_no = 0.0

    top1gt = 0
    top5gt = 0

    top1no = 0
    top5no = 0



    for i in range(len(image_list_plain)):

        print(f"Result of plain image {i} :")
        res, probs = predict(image_list_plain[i])
        result_gt += res
        if true_label == probs[0,0]:
            top1gt += 1
            top5gt += 1
        elif true_label in probs:
            top5gt += 1

        print(f"Result of noisy image {i} :")
        res_n, probs_n = predict(image_list_noisy[i])
        result_no += res_n
        if true_label == probs_n[0,0]:
            top1no += 1
            top5no += 1
        elif true_label in probs_n:
            top5no += 1

        print("\n")

    i += 1
    print(f"Avarage ground truth confidence = {str(result_gt.cpu().numpy() / i)}, Top1 Accuracy {str(top1gt / i)}, Top5 Accuracy {str(top5gt / i)} ")
    print(f"Avarage noisy confidence = {str(result_no.cpu().numpy() / i)}, Top1 Accuracy {str(top1no / i)}, Top5 Accuracy {str(top5no / i)} ")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="apple")
    parser.add_argument('--true_label_name', type=str, default="apple")
    parser.add_argument('--target_label_name', type=str, default="tree")
    parser.add_argument('--plain_folder', type=str, default="output/apple_plain/train/ours_30000/renders")
    parser.add_argument('--noisy_folder', type=str, default="output/apple_noisy/train/ours_30000/renders")
    parser.add_argument('--resolution', type=int, default=1600)
    parser.add_argument('--sam_ckpt_name', type=str, default="sam_vit_h_4b8939.pth")
    parser.add_argument('--clip_model', type=str, default="ViT-B/32")
    #['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    #dataset_path = os.path.join("data/" + args.dataset_name)
    sam_ckpt_path = os.path.join("sam_ckpt/" + args.sam_ckpt_name)
    img_folder_plain = args.plain_folder
    img_folder_noisy = args.noisy_folder

    data_list_plain = os.listdir(img_folder_plain)
    data_list_plain.sort()

    data_list_noisy = os.listdir(img_folder_noisy)
    data_list_noisy.sort()

    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    class_idx = json.load(urllib.request.urlopen(url))
    imagenet_classes = [class_idx[str(k)][1] for k in range(len(class_idx))]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model, transform = clip.load(args.clip_model, device="cuda:1") 

    true_label_name = args.true_label_name
    target_label_name = args.target_label_name

    true_label = find_or_add_label(true_label_name, imagenet_classes)
    target_label = find_or_add_label(target_label_name, imagenet_classes)


    text = clip.tokenize(imagenet_classes).to("cuda:1")

    #   CLIP's normalization parameters
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.8,
        box_nms_thresh=0.8,
        stability_score_thresh=0.9,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    img_list = []
    for data_path in data_list_plain:
        image_path = os.path.join(img_folder_plain, data_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = torch.from_numpy(image)
        img_list.append(image)

    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    imgs_plain = torch.cat(images)


    img_list = []
    for data_path in data_list_noisy:
        image_path = os.path.join(img_folder_noisy, data_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = torch.from_numpy(image)
        img_list.append(image)

    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    imgs_noisy= torch.cat(images)
    
    
    run(imgs_plain, imgs_noisy)
    
    