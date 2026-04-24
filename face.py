'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import torch

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: torch.Tensor) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    detection_results: List[List[float]] = []
    # print(img.shape)
    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    
    img_np = img.numpy().astype('uint8')
    locs = face_recognition.face_locations(img_np)
    
    for (top, right, bottom, left) in locs:
        x = float(left)
        y = float(top)
        w = float(right - left)
        h = float(bottom - top)
        # detection_results.append([x, y, w, h])
        new_h = h * 1.2
        new_y = max(0.0, y - (new_h - h) / 2)
        new_h = min(img.shape[0] - new_y, new_h)
        detection_results.append([x, new_y, w, new_h])

    return detection_results



def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    cluster_results: List[List[str]] = [[] for _ in range(K)] # Please make sure your output follows this data format.
    
    file_names = []
    fea_list = []
    
    for name, img in imgs.items():
        if img.shape[0] != 3:
            raise IndexError
        file_names.append(name)
        
        # extract features
        img = img.permute(1, 2, 0)
        img_np = img.numpy().astype('uint8')
        locs = face_recognition.face_locations(img_np)
        # 128 demension
        encodings = face_recognition.face_encodings(img_np, locs)
        fea = torch.tensor(encodings[0], dtype=torch.float32)
        fea_list.append(fea)
    
    # [num, 128]
    fea_torch = torch.stack(fea_list)
    num = len(fea_list)
    
    rand_idx = torch.randperm(num)[:K]
    center_ids = fea_torch[rand_idx].clone()
    prev_labels = None
    labels = torch.zeros(num, dtype=torch.long)
    
    for _ in range(1000):
        distances = torch.cdist(fea_torch, center_ids)
        labels = torch.argmin(distances, dim=1)
        if prev_labels != None and torch.equal(labels, prev_labels):
            break
        prev_labels = labels.clone()
        
        for i in range(K):
            samples = (labels == i)
            if samples.sum() > 0:
                center_ids[i] = fea_torch[samples].mean(dim=0)
    
    for i, name in enumerate(file_names):
        cluster_results[int(labels[i].item())].append(name)
    
    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)
# import json
# import os
# from PIL import Image, ImageDraw

# def visualize_predictions(json_path, img_dir, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
    
#     with open(json_path, 'r') as f:
#         results = json.load(f)

#     for img_name, bboxes in results.items():
#         img_path = os.path.join(img_dir, img_name)
#         if not os.path.exists(img_path):
#             continue
            
#         img = Image.open(img_path).convert('RGB')
#         draw = ImageDraw.Draw(img)
        
#         for bbox in bboxes:
#             x, y, w, h = bbox
#             draw.rectangle([x, y, x + w, y + h], outline="gray", width=3)
            
#         img.save(os.path.join(output_dir, img_name))
#     print(f"saved to: {output_dir}")

# visualize_predictions('result_task1_val.json', 'validation_folder/images', 'debug_folder')

# import json
# import os
# from PIL import Image, ImageDraw
# from collections import defaultdict

# def visualize_ground_truth(gt_json_path, img_dir, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
    
#     with open(gt_json_path, 'r') as f:
#         gt_data = json.load(f)
    
#     gt_dict = defaultdict(list)
#     for item in gt_data:
#         gt_dict[item['iname']].append(item['bbox'])

#     for img_name, bboxes in gt_dict.items():
#         img_path = os.path.join(img_dir, img_name)
#         if not os.path.exists(img_path):
#             continue
            
#         img = Image.open(img_path).convert('RGB')
#         draw = ImageDraw.Draw(img)
        
#         for bbox in bboxes:
#             x, y, w, h = bbox
#             draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
            
#         img.save(os.path.join(output_dir, img_name))
#     print(f"saved to: {output_dir}")

# visualize_ground_truth('validation_folder/ground-truth.json', 'validation_folder/images', 'debug_validation')