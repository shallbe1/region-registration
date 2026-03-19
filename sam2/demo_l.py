import cv2
from pathlib import Path
import os
import torch
import sys
sys.path.append("/media/dell/data/zhangyc/regional_registration")# need change

from gluefactory.utils.image import ImagePreprocessor
from gluefactory.utils.tensor import batch_to_device
from gluefactory.eval.io import load_model
from omegaconf import OmegaConf
import numpy
import matplotlib.pyplot as plt
from PIL import Image
from gluefactory.visualization.viz2d import plot_images, plot_keypoints, plot_matches
from copy import deepcopy

from scipy.ndimage import distance_transform_edt,zoom
from scipy import stats

from kmeans_pytorch import kmeans
import kornia
import kornia.utils as KU
import torch.nn.functional as F
from demo_sam1 import SAM2,build_sam
from collections import defaultdict
from scipy.ndimage import label as scipy_label
from skimage import measure

'''
Part of the project code is based on LightGlue and SAM2, and we are grateful for their team's outstanding contributions. Thanks to Tangfei Liao for helping with the LightGlue output debugging. This code can be found in github.  
'''

def read_image(path, grayscale: bool = False):
    """Read an image from path as RGB or grayscale"""
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def numpy_image_to_torch(image):
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def load_image(path, grayscale=False):
    image = read_image(path, grayscale=grayscale)
    return numpy_image_to_torch(image)


def tocuda(data):
    for key in data.keys():
        if type(data[key]) == torch.Tensor:
            data[key] = data[key].cuda().unsqueeze(0)
        elif type(data[key]) == numpy.ndarray:
            data[key] = torch.from_numpy(data[key])
            data[key] = data[key].cuda().unsqueeze(0)

    return data


def homo(image,height,width,H,device):

    src = image.unsqueeze(0).permute(0,3,1,2)#.to(device)#.permute(2, 1, 0).unsqueeze(0) 
    #H=numpy.linalg.inv(H)
    H = torch.tensor(H,dtype=torch.float32).to(device)
    
    grid = KU.create_meshgrid(height, width, normalized_coordinates=False).to(device) 
    
    grid_homogeneous = torch.cat([grid, torch.ones_like(grid[...,:1])], dim=-1) 
    
    new_grid_homogeneous = torch.matmul(H, grid_homogeneous.view(-1,3).T) 
    
    new_grid_homogeneous = new_grid_homogeneous.T 
    new_grid_homogeneous = new_grid_homogeneous.reshape(1, height, width, 3) 
    new_grid_x = new_grid_homogeneous[..., 0]/new_grid_homogeneous[..., 2]
    new_grid_y = new_grid_homogeneous[..., 1]/new_grid_homogeneous[..., 2]
    new_grid = torch.stack([new_grid_x, new_grid_y], dim=-1) 
    
    return new_grid-grid,grid
    
def mesh (image,height,width,grid):
    src = image.unsqueeze(0).permute(0,3,1,2)
    new_grid_normalized = 2.0 * grid / torch.tensor([width - 1, height - 1], dtype=torch.float32, device=device) - 1.0
    output_image = F.grid_sample(src, new_grid_normalized, mode='bilinear', padding_mode='zeros', align_corners=False)
    output_image = output_image.squeeze(0).permute(1, 2, 0)#.detach().cpu().numpy()  
    output_image = torch.clamp(output_image * 255, 0, 255)
    #output_image = numpy.clip(output_image * 255, 0, 255).astype(numpy.uint8)
    
    return output_image
    
def mesh1 (image,height,width,grid):
    src = image.unsqueeze(0).permute(0,3,1,2)
    new_grid_normalized = 2.0 * grid / torch.tensor([width - 1, height - 1], dtype=torch.float32, device=device) - 1.0
    output_image = F.grid_sample(src, new_grid_normalized, mode='bilinear', padding_mode='zeros', align_corners=False)
    output_image = output_image.squeeze(0).permute(1, 2, 0)#.detach().cpu().numpy()  
    
    #output_image = numpy.clip(output_image * 255, 0, 255).astype(numpy.uint8)
    
    return output_image    


    
from skimage.transform import PiecewiseAffineTransform, warp

def non_rigid_stitching(image1, image2, points1, points2):
    
    transform = PiecewiseAffineTransform()
    transform.estimate(points1, points2)
    
    warped_image2 = warp(image2, transform, output_shape=image1.shape)
    
    return warped_image2       

def mask_choose(use_matrix,img_count_k_r,mask):
    mask[(img_count_k_r-use_matrix)<0]=0
    return mask
def seg_label(anns,ky1,labels,h1,w1,k,device):
    labels=labels.to(device)
    if len(anns) == 0:
        return
    
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    shape_0=sorted_anns[0]['segmentation'].shape[0]
    shape_1=sorted_anns[0]['segmentation'].shape[1]
    img = torch.zeros((shape_0, shape_1, 1),device=device,dtype=torch.float32)
    img_count = torch.zeros_like(img)
    mask_nolabel = torch.ones_like(img)
    
    segmentation_tensors=[torch.tensor(ann['segmentation'],device=device,dtype=torch.bool)
                          for ann in sorted_anns]
    
    
    for number,m in enumerate(segmentation_tensors):
        img[m] = number+1
    x_coords=ky1[:,0].long()
    y_coords=ky1[:,1].long()
    
    region_labels=img[y_coords,x_coords,0]
    count_matrix=torch.zeros((number+1,k),device=device,dtype=torch.int32)
    #numpy.add.at(count_matrix,(A-1,B-1),1)
    
    for i in range(1,number+1):
        for j in range(1,k+1):
            count_matrix[i-1,j-1]=torch.sum((region_labels==i)&(labels==j-1))#20*5
    
    max_value=torch.argmax(count_matrix,dim=1)+1# +1
    is_zero_raw=torch.all(count_matrix==0,dim=1)
    max_value[is_zero_raw] = k+2#6##not choose
    max_value=max_value.float()
    
    
    for number_x,m in enumerate(segmentation_tensors):
        img_count[m] = max_value[number_x]
    
    img_count_copy1=deepcopy(img_count)
    kernel = numpy.ones((3,3),numpy.uint8)#cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    
    
    for number_x,ann_x in enumerate(segmentation_tensors):
        if max_value[number_x]==k+2: 
            ann_num=torch.sum(ann_x)
            for num_x in range(1,k+1):
                
                iou=(torch.sum(ann_x&(img_count.squeeze()==num_x))/ann_num).item()
                if iou>0.5:
                    break
            
            if iou>0.5:
                img_count[ann_x]=num_x    
                img_count_copy1[ann_x]=num_x
            else:
                
                expand_mask=torch.tensor(cv2.dilate(ann_x.cpu().numpy().astype(numpy.uint8),kernel),
                    device=device).bool()
                expand_region_mask = expand_mask&~(ann_x.bool())
                masked_image=img_count*expand_region_mask.unsqueeze(-1)
                
                expanded_colors=masked_image[expand_region_mask]
                try:
                    unique_values=torch.bincount(expanded_colors.flatten().long())
                    unique_values=torch.argsort(unique_values,descending=True)[:2]
                    if (unique_values[0]!=0)&(unique_values[0] != k+2):
                        color=unique_values[0]
                    else:
                        color=unique_values[1]
                #color=unique_values[(unique_values!=0)&(unique_values != k+2)]
                
                except:
                    unique_values=torch.bincount(expanded_colors.flatten().long())
                    color=unique_values.argmax()
            #most_color=stats.mode(expanded_colors.reshape(-1,1),axis=0)
            
                if (color!=0)&(color != k+2):
                    img_count[ann_x] = color.float()
    
    
    kernel1=numpy.ones((5,5),numpy.uint8)

    img_count=img_count.cpu().numpy()
    img_count_copy=deepcopy(img_count)
    
    erode_image=numpy.expand_dims(cv2.dilate(img_count,kernel1,iterations=3),axis=-1)
    
    maskk=(img_count==0) | (img_count==k+2)
    img_count_copy[maskk]=erode_image[maskk]
    
    num_labels_c, labels_c = cv2.connectedComponents(((img_count_copy==0)|(img_count_copy==k+2)).astype(numpy.uint8)) 
    
    
    img_count=torch.tensor(img_count,device=device)
    img_count_copy=torch.tensor(img_count_copy,device=device)
    labels_c=torch.tensor(labels_c,device=device)
    for label_c in range(1, num_labels_c): 
        region_mask_c = (labels_c == label_c) 
        edges = torch.tensor(cv2.dilate(region_mask_c.cpu().numpy().astype(numpy.uint8), kernel),device=device) & ~(region_mask_c.bool())
        boundary_indices = torch.where(edges != 0)
        boundary_colors = img_count[boundary_indices]
        
        try:
            unique_values=torch.bincount(boundary_colors.flatten().long())
            unique_values=torch.argsort(unique_values,descending=True)[:2]
            if (unique_values[0]!=0)&(unique_values[0] != k+2):
                color_c=unique_values[0]
            else:
                color_c=unique_values[1]
        except:
             unique_values=torch.bincount(expanded_colors.flatten().long())
             color_c=unique_values[0]
        img_count_copy[region_mask_c] = color_c.float()
    
    
    
    #return torch.tensor(img_count_copy,device=device),torch.tensor(img_count_copy1,device=device)
    
    return img_count_copy,img_count_copy1


if __name__ == '__main__':
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask_generator=build_sam()
    
    region_size=4
    view0_p = '../assets/19.jpg'
    view1_p = '../assets/19_1.jpg'
    img_preprocessor = ImagePreprocessor(OmegaConf.create(
        {"resize": 480, "edge_divisible_by": None, "side": 'short', "interpolation": 'bilinear',
         "align_corners": None, "antialias": True, "square_pad": False, "add_padding_mask": False}
    ))

    view0 = load_image(view0_p)
    view1 = load_image(view1_p)
    view0_t, view1_t = deepcopy(view0), deepcopy(view1)
    view0, view1 = img_preprocessor(view0), img_preprocessor(view1)
    data = {
        "view0": tocuda(view0), "view1": tocuda(view1)
    }

    model_conf = {
        "name": 'two_view_pipeline',
        "ground_truth": {"name": None},
        "extractor": {"name": 'gluefactory_nonfree.superpoint',
                      "max_num_keypoints": 2048, "detection_threshold": 0.0, "nms_radius": 3},
        "matcher": {"name": 'matchers.lightglue_pretrained', "features": 'superpoint',
                    "depth_confidence": -1, "width_confidence": -1, "filter_threshold": 0.1}
    }

    model = load_model(OmegaConf.create(model_conf), None)
    model = model.to(device).eval()
    
    pred = model(batch_to_device(data, device, non_blocking=True))
    
    kernel = numpy.ones((3,3),numpy.uint8)
    
    # renormalization
    for k in pred.keys():
        #print(k)
        if k.startswith("keypoints"):
            idx = k.replace("keypoints", "")
            scales = 1.0 / (
                data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
            )
            pred[k] = pred[k] * scales[None]
     
    pts0, pts1 = pred['keypoints0'].squeeze(0), pred['keypoints1'].squeeze(0)
    use_temperature=0
    k=3 #Number of clusters needs to be adjusted manually.
        
    score0=pred['scores'][0]>0.2
    
    ky0 = pts0[pred['matches'][0][score0][:, 0]]
    ky1 =pts1[pred['matches'][0][score0][:, 1]]
    
    imgg0=cv2.imread(view0_p)
    imgg1=cv2.imread(view1_p)
    
    #H,sta = cv2.findHomography(ky0.cpu().numpy(),ky1.cpu().numpy(),cv2.RANSAC,5.0)
    h1,w1,c1=imgg0.shape
    
    image = torch.tensor(imgg0).float() / 255.0
    image1 = torch.tensor(imgg1).float() / 255.0
    img_cuda=image.to(device)
    device1=device
    
    src = deepcopy(image).permute(2, 1, 0).unsqueeze(0) 
    vectors=ky1-ky0
    meshgrid= torch.stack(torch.meshgrid(torch.arange(0, h1), torch.arange(0, w1)), dim=-1).float()
    meshgrid = meshgrid.unsqueeze(0)#ssss.unsqueeze(0)  # shape: [1, 1, H, W, 2]
    meshgrid = meshgrid / torch.tensor([h1, w1]).float() * 2.0 - 1.0  
    
    
    labels,_ = kmeans(X=vectors,num_clusters=k,distance='cosine',device=device1)
    lengths = torch.norm(vectors,dim=1)
    #print(lengths,"-----")
    average_lengths=[]
    for label_k in range(k):
        average_length=lengths[labels==label_k].mean()
        average_lengths.append(average_length)
    
    length_k=torch.argsort(torch.tensor(average_lengths))    
    
    auto_mask=SAM2(imgg0,mask_generator)
    
    
    sam_map,img_count=seg_label(auto_mask,ky0,labels,h1,w1,k,device1)
    
    ky0=ky0.cpu()
    ky1=ky1.cpu()
    
    
    mask_overlap=[]
    combine_image=torch.zeros((h1,w1,3),device=device1)
    overlap_image=[]
    temperatures=[]
    mask_3ds=[]
   
    imgg1_gray=cv2.cvtColor(imgg1,cv2.COLOR_BGR2GRAY)
    #for k_new in k_new_index:
    use_matrix=numpy.zeros((h1,w1,1))
    
    for i in length_k:
    
        i=i.item()
        cluster_points0 = ky0[labels == i]  
        cluster_points1 = ky1[labels == i]
        
        H_k,sta_k = cv2.findHomography(cluster_points1.numpy(),cluster_points0.numpy(),cv2.RANSAC,5.0)
        
        flow,grid =homo(img_cuda,h1,w1,H_k,device1)#
        sam_map_copy=sam_map.clone()
        sam_map_copy[sam_map!=i+1]=0
        mask_k=mesh(sam_map_copy,h1,w1,
                      grid+flow).to(torch.bool)#(numpy.uint8)#mask move
        
        img_count_copy1=img_count.clone()
        img_count_copy1[sam_map!=i+1]=0
        img_count_k = mesh1(img_count_copy1,h1,w1,
                      grid+flow)
                      
        img_count_k_r = img_count_k.clone()
        img_count_k_r[img_count_k_r!=i+1]=0
        mask_return=mask_choose(use_matrix,img_count_k_r.cpu().numpy(),mask_k.cpu().numpy())
        #mask_return=mask_choose(use_matrix,img_count_k_r,mask_k)
        
        use_matrix=((use_matrix+mask_return)>0).astype(bool).astype(numpy.uint8)
        
        mask_overlap.append(mask_k.cpu().numpy().astype(numpy.uint8))
        if use_temperature==1:
            temperature=numpy.mean(imgg1_gray[img_count_k.astype(numpy.uint8)])
            temperatures.append(temperature)
        
        mask_k=mask_return####use temperature delete this 
        mask_3d=torch.tensor(numpy.repeat(mask_k,3,axis=-1),device=device1) 
        img_k =mesh(img_cuda,h1,w1,grid+flow)
        
        combine_image[mask_3d]=img_k[mask_3d]
        
        mask_3ds.append(mask_3d.cpu().numpy())
        overlap_image.append(img_k.cpu().numpy())
        
    mask_overlaps=numpy.stack(mask_overlap,axis=0)
    overlaps_empty=numpy.sum(mask_overlaps,axis=0)
    
    index_tems=numpy.argsort(temperatures)
    
    combine_image=combine_image.cpu().numpy()
    if use_temperature==1:
        combine_image=numpy.zeros((h1,w1,3))
        for index_tem in index_tems:
        
            combine_image[mask_3ds[index_tem]]=overlap_image[index_tem][mask_3ds[index_tem]]
    
    
    combine_image=combine_image.astype(numpy.uint8)
    mask_inpaint=numpy.zeros_like(overlaps_empty)
    
    mask_inpaint[overlaps_empty==0]=255
    fill_image=cv2.inpaint(combine_image,mask_inpaint.astype(numpy.uint8),3,cv2.INPAINT_TELEA)
    
    combine_image = cv2.cvtColor(combine_image, cv2.COLOR_BGR2RGB) 
    fill_image = cv2.cvtColor(fill_image, cv2.COLOR_BGR2RGB) 
    
    blended_k = cv2.addWeighted(combine_image,0.5,imgg1,0.5,0)
    plt.subplot(2,2,1)
    plt.imshow(blended_k)
    plt.subplot(2,2,2)
    plt.imshow(img_count.cpu().numpy())
    plt.subplot(2,2,3)
    plt.imshow(overlaps_empty)
    plt.subplot(2,2,4)
    plt.imshow(fill_image)
    plt.show()   
    
    
