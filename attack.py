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
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image

def segment(indx, image):

    #aug_imgs = torch.cat([image])
    seg_images, seg_image_masks, seg_map = sam_encoder(image)

    for i, img in tqdm(enumerate(seg_images)):
        
        image = to_pil_image(img)
        #res = predict(image)

        image = image * seg_image_masks[i][:, :, None]
        image = to_pil_image(image)                       
        #image.save(save_folder + "/segmented_" + str(i) + "_" + data_list[indx])

    seg_images_ = [seg_images[i].transpose(2, 0, 1)[None, ...] for i in range(len(seg_images))]
    #seg_images_torch = torch.from_numpy(seg_images_)

    return seg_images, seg_image_masks, seg_map

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

def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    seg_img_mask = mask['segmentation'][y:y+h, x:x+w, ...]
    return seg_img, seg_img_mask

def get_square_seg_img(mask, image):
    image = image.copy()
    #image[mask['segmentation'] == 0] = np.array([0, 0, 0], dtype=np.uint8)
    
    # Original bounding box
    x, y, w, h = np.int32(mask['bbox'])
    
    # Determine the longer edge
    longer_edge = max(w, h)
    
    # Calculate center of the original bounding box
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Adjust bounding box to be square based on the longer edge
    x_new = max(center_x - longer_edge // 2, 0)
    y_new = max(center_y - longer_edge // 2, 0)
    
    # Ensure the new bounding box does not exceed the image boundaries
    x_new = min(x_new, image.shape[1] - longer_edge)
    y_new = min(y_new, image.shape[0] - longer_edge)
    
    # Crop the image and mask using the adjusted bounding box
    seg_img = image[y_new:y_new + longer_edge, x_new:x_new + longer_edge, ...]
    seg_img_mask = mask['segmentation'][y_new:y_new + longer_edge, x_new:x_new + longer_edge, ...]
    mask['bbox'] = (x_new, y_new, longer_edge, longer_edge)

    return seg_img, seg_img_mask

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def sam_encoder(image):
    image = image.permute(1,2,0).numpy().astype(np.uint8)

    masks = mask_generator.generate(image)

    # pre-compute postprocess
    masks = masks_update(masks, iou_thr=0.9, score_thr=0.8, inner_thr=0.7)
    
    def mask2segmap(masks, image):
        seg_img_list = []
        seg_img_masks = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks[0])):
            mask = masks[0][i]
            pad_seg_img, seg_img_mask = get_square_seg_img(mask, image)
            
            #pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
            #predict(pad_seg_img.transpose(2,0,1))

            #pad_seg_img = cv2.resize(pad_seg_img, (224,224))
            seg_img_list.append(pad_seg_img)
            seg_img_masks.append(seg_img_mask)

            seg_map[masks[0][i]['segmentation']] = i
        #seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
        #seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to(device)

        return seg_img_list, seg_img_masks, masks[0]

    seg_images, seg_maps = (), ()
    seg_images, seg_image_masks, seg_maps = mask2segmap(masks, image)


    return seg_images, seg_image_masks, seg_maps

def find_or_add_label(label, class_list):
    if label in class_list:
        return class_list.index(label)
    else:
        class_list.append(label)
        return len(class_list) - 1

def clip_values(image):
    clamp_ranges = {
        'min': torch.tensor([-1.79226253374815, -1.7520971281645974, -1.4802197687835659]), # Minimum values for each channel
        'max': torch.tensor([1.930336254158794, 2.0748838377332515, 2.1458969890575763])  # Maximum values for each channel
    }

    # Ensure the clamp ranges are the same device and dtype as the adversarial_image
    clamp_ranges['min'] = clamp_ranges['min'].to(image.device, dtype=image.dtype)
    clamp_ranges['max'] = clamp_ranges['max'].to(image.device, dtype=image.dtype)

    # Manually apply clamping for each channel
    for c in range(image.shape[1]): # Iterate over the channel dimension
        image[:, c, :, :] = torch.clamp(image[:, c, :, :], clamp_ranges['min'][c], clamp_ranges['max'][c])

    return image

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def unnormalize(im):
    tensor = torch.tensor(im).float().cpu()
    tensor = tensor.clone().detach()
    tensor.mul_(std).add_(mean)
    tensor = tensor.cpu().squeeze()
    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
    return tensor

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

    return res, image

def generate_noise(indx, image_orj, seg_images, seg_image_masks, seg_map, learning_rate= 0.02, it=30):

    print("Orijinal Image Prediction : ")
    #orj_im_pil = to_pil_image(image_orj)
    orj_res, orj_im = predict(image_orj)
    if orj_res < 0.05:
        print(f"Confidence:{orj_res} is below threashold for input image, skipping...")
        to_pil_image(image_orj).save(save_folder + "/" + data_list[indx])
        return

    best_pred = 0
    for i, img in tqdm(enumerate(seg_images)):
        img_masked = img * seg_image_masks[i][:,:,None]
        pred, im = predict(img_masked)
        if pred > best_pred:
            best_pred = pred
            best_mask = seg_map[i]
            best_image_id = i
            best_image = img
    print(f"Best segmented image ID:{best_image_id}, Confidence:{best_pred}")
    #to_pil_image(best_image).save(save_folder + "/best_segment_" + data_list[indx])

    image = to_pil_image(image_orj)

    image = transform(image).unsqueeze(0).to("cuda:1")
    image_original = image.clone()
    image_original_np = image_original.detach().cpu().numpy()
    criterion = torch.nn.CrossEntropyLoss()

    noise = torch.empty_like(image)

    seg_image_mask = cv2.resize(seg_map[best_image_id]["segmentation"].astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)
    inverse_image_original_np = image_original_np * np.logical_not(seg_image_mask[None, None, :, :])
    

    for step in range(it):
        image.requires_grad = True

        outputs, _ = model(image, text)
        probs = outputs.softmax(dim=-1).detach().cpu().numpy()

        loss = criterion(outputs, torch.tensor([true_label], device="cuda:1"))
        model.zero_grad()
        loss.backward()

        print(f"Step={step}, Loss={loss.item()}" + " True label[" + true_label_name + "]="+ str(probs[0,true_label]) + " Target label["+ target_label_name + "]="+ str(probs[0,target_label]))
  
        if probs[0,true_label] == 0.0 and loss.item() > 20.00:
            break

        with torch.no_grad():
            image += learning_rate * image.grad.sign()
            noise += learning_rate * image.grad.sign()

        image = torch.from_numpy(image.clone().detach().cpu().numpy() * seg_image_mask[None, None, :, :] + inverse_image_original_np).to("cuda:1")
        image = clip_values(image)

    image = unnormalize(image)
    print("Raw Image Prediction : ")
    predict(image)
    #to_pil_image(image).save(save_folder + "/raw_" + data_list[indx])

    resize_transform = transforms.Resize((image_orj.shape[1], image_orj.shape[2]))
    resized_image = resize_transform(image.unsqueeze(0)).squeeze(0)

    #resized_image = image.resize((image_orj.shape[1], image_orj.shape[2]), Image.BICUBIC)
    print("Resized Image Prediction : ")
    predict(resized_image)
    #to_pil_image(resized_image).save(save_folder + "/resized_" + data_list[indx])
    
    merged_img = merge(image_orj, resized_image, best_mask)
    
    print("Noisy Image Prediction : ")
    predict(merged_img)
    to_pil_image(merged_img).save(save_folder + "/" + data_list[indx])

    noise = unnormalize(noise)
    noise *= seg_image_mask[None, :, :]
    noise = resize_transform(noise.unsqueeze(0)).squeeze(0)
    #to_pil_image(noise).save(save_folder + "/noise_" + data_list[indx])

    merged_orj_im = (image_orj + noise).clamp(0, 255)
    predict(merged_orj_im)
    #to_pil_image(merged_orj_im).save(save_folder + "/merged_orj_im_" + data_list[indx])

    print("End of noise generation.")

def merge(image_orj, resized_image, seg_map):

    masked_image_tensor = resized_image * seg_map["segmentation"][None, :, :]

    inverse_image_orj = image_orj * np.logical_not(seg_map["segmentation"][None, :, :])

    inverse_image_orj += masked_image_tensor

    return inverse_image_orj

def run(image_list):
 
    mask_generator.predictor.model.to(device)

    for i, img in tqdm(enumerate(image_list), leave=False):

        seg_images, seg_image_masks, seg_map = segment(i, img)

        generate_noise(i, img, seg_images, seg_image_masks, seg_map)
     
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="apple")
    parser.add_argument('--true_label_name', type=str, default="apple")
    parser.add_argument('--target_label_name', type=str, default="stop sign")
    parser.add_argument('--resolution', type=int, default=1600)
    parser.add_argument('--sam_ckpt_name', type=str, default="sam_vit_h_4b8939.pth")
    parser.add_argument('--clip_model', type=str, default="ViT-B/32")
    #['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = os.path.join("data/" + args.dataset_name)
    sam_ckpt_path = os.path.join("sam_ckpt/" + args.sam_ckpt_name)
    img_folder = os.path.join(dataset_path, 'images')
    data_list_raw = os.listdir(img_folder)
    data_list_raw.sort()

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
    mod  = 1
    data_list = []

    for i, data_path in tqdm(enumerate(data_list_raw), leave=False):
        if(i%mod != 0):
            continue

        data_list.append(data_path)  
              
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        aspect_ratio = h / w
        new_h = int(args.resolution * aspect_ratio)

        if aspect_ratio != 1:
            print("Aspect ratio of input image is not 1. Adjusting input.. Height : " + str(h) + "  Width :" + str(w))
            new_size = min(w, h)
            image = image[0:new_size, 0:new_size]

        image = torch.from_numpy(image)
        img_list.append(image)

    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    imgs = torch.cat(images)

    save_folder = os.path.join(dataset_path, 'images_noisy')
    os.makedirs(save_folder, exist_ok=True)
    
    run(imgs)
    