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
        image = transform(image).unsqueeze(0).to("cuda:1")
        outputs, _ = model(image, text)
        probs = outputs.softmax(dim=-1)

        res = probs[0,true_label]

        top5_probs, top5_inds = probs.topk(5, dim=-1)

        top5_probs = top5_probs.detach().cpu().numpy()
        top5_inds = top5_inds.detach().cpu().numpy()

        print(f"True label : {imagenet_classes[true_label]}, {res:.5f} Pred 1: {imagenet_classes[top5_inds[0, 0]]}, {top5_probs[0, 0]:.5f} 2: {imagenet_classes[top5_inds[0, 1]]}, {top5_probs[0, 1]:.5f} 3: {imagenet_classes[top5_inds[0, 2]]}, {top5_probs[0, 2]:.5f} 4: {imagenet_classes[top5_inds[0, 3]]}, {top5_probs[0, 3]:.5f} 5: {imagenet_classes[top5_inds[0, 4]]}, {top5_probs[0, 4]:.5f}")


    return res

def segment(indx, image, sam_encoder):

    aug_imgs = torch.cat([image])
    seg_images, seg_image_masks, seg_map = sam_encoder(aug_imgs)

    for i, img in tqdm(enumerate(seg_images)):
        
        image = to_pil_image(img)
        res = predict(image)

        image = image * seg_image_masks[i][:, :, None]
        image = to_pil_image(image)                       
        #image.save(save_folder + "/segmented_" + str(i) + "_" + data_list[indx])

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
    image = cv2.cvtColor(image[0].permute(1,2,0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
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

    # 0:default 1:s 2:m 3:l
    return seg_images, seg_image_masks, seg_maps

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

def generate_noise(indx, image_orj, seg_images, seg_image_masks, seg_map, learning_rate= 0.1, it=100):

    best_pred = 0
    for i, img in tqdm(enumerate(seg_images)):
        img_masked = img * seg_image_masks[i][:,:,None]
        pred = predict(to_pil_image(img_masked))
        if pred > best_pred:
            best_pred = pred
            best_mask = seg_map[i]
            best_image_id = i
            best_image = img
    print(f"Best segmented image ID:{best_image_id}, Confidence:{best_pred}")

    image_orj_mem = cv2.cvtColor(image_orj.permute(1,2,0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    image = to_pil_image(image_orj_mem)
    image = transform(image).unsqueeze(0).to("cuda:1")
    image_original = image.clone()
    image_original_np = image_original.detach().cpu().numpy()
    criterion = torch.nn.CrossEntropyLoss()

    noise = torch.empty_like(image)

    seg_image_mask = cv2.resize(seg_map[0]["segmentation"].astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)
    inverse_image_original_np = image_original_np * np.logical_not(seg_image_mask[None, None, :, :])
    

    for step in range(it):
        image.requires_grad = True

        outputs, _ = model(image, text)
        probs = outputs.softmax(dim=-1).detach().cpu().numpy()

        loss = criterion(outputs, torch.tensor([true_label], device="cuda:1"))
        model.zero_grad()
        loss.backward()

        
        print(f"Step={step}, Loss={loss.item()}" + " True label[" + true_label_name + "]="+ str(probs[0,true_label]) + " Target label["+ target_label_name + "]="+ str(probs[0,target_label]))
  
        with torch.no_grad():
            image += learning_rate * image.grad.sign()
            noise += learning_rate * image.grad.sign()

        image = torch.from_numpy(image.clone().detach().cpu().numpy() * seg_image_mask[None, None, :, :] + inverse_image_original_np).to("cuda:1")
        image = clip_values(image)

    un_image = image.clone().detach()

    image = to_pil_image(unnormalize(un_image).permute(2,0,1))

    predict(image)

    resized_image = image.resize((image_orj_mem.shape[0], image_orj_mem.shape[1]), Image.BICUBIC)


    resized_image.save(save_folder + "/resized_" + data_list[indx])
    print("Resized Image Prediction : ")
    predict(resized_image)

    merged_img = merge(image_orj, resized_image, seg_map[0]["segmentation"])
    
    print("Orijinal Image Prediction : ")
    predict(to_pil_image(image_orj))

    print("Noisy Image Prediction : ")
    predict(merged_img)

    merged_img.save(save_folder + "/" + data_list[indx])

    noise = noise.clone().detach().cpu().numpy() * seg_image_mask[None, None, :, :]
    noise = unnormalize(noise).permute(2,0,1)
    noise = to_pil_image(noise)
    noise.save(save_folder + "/noise_" + data_list[indx])

    noise_crop, _ = get_seg_img(seg_map[0], np.array(merged_img))
    noise_crop = to_pil_image(noise_crop)
    noise_crop = noise_crop.resize((1080,1080), Image.NEAREST)
    noise_crop.save(save_folder + "/noisecrop_" + data_list[indx])



def merge(image_orj, resized_image, seg_map):

    resized_image_tensor = torch.from_numpy(np.array(resized_image)).permute(2,0,1)  # From HWC to CHW and to float
    masked_image_tensor = resized_image_tensor.clone().detach().cpu().numpy() * seg_map[None, :, :]
    
    
    image_orj_rgb = image_orj[[2, 1, 0], :, :]
    inverse_image_orj = image_orj_rgb * np.logical_not(seg_map[None, :, :])

    #x, y, w, h = seg_map['bbox']

    inverse_image_orj += masked_image_tensor

    return to_pil_image(inverse_image_orj)

def run(image_list):
 
    mask_generator.predictor.model.to(device)

    for i, img in tqdm(enumerate(image_list), leave=False):

        seg_images, seg_image_masks, seg_map = segment(i, img.unsqueeze(0), sam_encoder)

        generate_noise(i, img, seg_images, seg_image_masks, seg_map)


def test(image_list):

    for i, img in tqdm(enumerate(image_list), leave=False):

        image = to_pil_image(img)
        print(f"Result of {data_list[i]} :")
        res = predict(image)
        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="bed")
    parser.add_argument('--true_label_name', type=str, default="banana")
    parser.add_argument('--target_label_name', type=str, default="window")
    parser.add_argument('--resolution', type=int, default=1600)
    parser.add_argument('--sam_ckpt_name', type=str, default="sam_vit_h_4b8939.pth")
    parser.add_argument('--clip_model', type=str, default="RN50")
    #['RN50', 'RN101', 'RN50x4', 'RN50x16',  'ViT-B/32', 'ViT-B/16']
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = os.path.join("data/" + args.dataset_name)
    sam_ckpt_path = os.path.join("sam_ckpt/" + args.sam_ckpt_name)
    img_folder = os.path.join(dataset_path, 'images')
    data_list = os.listdir(img_folder)
    data_list.sort()

    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    class_idx = json.load(urllib.request.urlopen(url))
    #imagenet_classes = [class_idx[str(k)][1] for k in range(len(class_idx))]
    imagenet_classes = [
            "door", "window", "stair", "wall", "roof",
            "car", "tree", "desk", "chair",
            "cup", "person", "lamp", "bridge", 
            "bottle", "bag", "clock", "hat", "glasses",
            "bench", "bicycle", "motorcycle", "bus", "truck",
            "traffic light", "mailbox", "bird",
            "fence", "sign", "parking meter", "sidewalk", "road",
            "garden", "pool", "playground"
        ]
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

    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)

        h, w = image.shape[:2]
        aspect_ratio = h / w
        new_h = int(args.resolution * aspect_ratio)

        image = cv2.resize(image, (args.resolution, new_h))
        image = torch.from_numpy(image)
        img_list.append(image)

    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    imgs = torch.cat(images)

    save_folder = os.path.join(dataset_path, 'images_noisy')
    os.makedirs(save_folder, exist_ok=True)
    
    run(imgs)
    
    #test(imgs)
    