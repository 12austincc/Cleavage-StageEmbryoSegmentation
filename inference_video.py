import json
import os
import random

import cv2
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import torch
from box import Box
from model import Model
from predictor import ModelPredictor
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
from tqdm import tqdm
import random
from PIL import Image
from resToCoco import segmentationToCocoResult
from ultralytics import YOLO
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as shapelyPolygon
from pycocotools import mask as maskUtils
from skimage import measure

def calculate_polygon_area(points):
    # Shoelace formula to calculate polygon area
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - y1 * x2
    area = abs(area) / 2.0
    return area

def drawAnns(img, anns, drawBbox, drawDes, paintBoundary,isGT):
    
    img_path = img
    I =Image.open(img) 
    if I.mode != 'RGB':
        I = I.convert('RGB')
    plt.imshow(I)

    polygons = []
    bbox_polygon = []
    bound_polygons = []
    color = []
    count_cell = 0

    blastomere_area = []
    fragment_total_area = 0
    cat2_positions = np.zeros((3, 3))
    for ann in anns:
        ax = plt.gca()
        ax.set_autoscale_on(False)
        # c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        if ann['category_id'] == 1:
            c = [1,0,0]
            count_cell += 1
        elif ann['category_id'] == 2:
            c = [0,1,0]
        if 'segmentation' in ann: # for gt
            if type(ann['segmentation']) == list:
                # polygon
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                    polygons.append(Polygon(poly))
                    bound_polygons.append(Polygon(poly))
                    color.append(c)
                    area = calculate_polygon_area(poly)
                    if ann['category_id'] == 1:
                        blastomere_area.append(area)
                    else:
                        fragment_total_area += area
                        # Calculate position in 3x3 grid
                        h, w = I.size
                        grid_h, grid_w = h // 3, w // 3
                        for i in range(3):
                            for j in range(3):
                                if np.any((poly[:, 0] >= j * grid_w) & (poly[:, 0] < (j + 1) * grid_w) &
                                          (poly[:, 1] >= i * grid_h) & (poly[:, 1] < (i + 1) * grid_h)):
                                    cat2_positions[i, j] += 1
                if drawBbox:
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y]]
                    np_poly = np.array(poly).reshape((4, 2))
                    polygons.append(Polygon(np_poly))
                    color.append(c)
            else:
                # mask for res
                t = img
                if type(ann['segmentation']['counts']) == list:
                    rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                else:
                    rle = [ann['segmentation']]
                m = maskUtils.decode(rle)

                if len(m.shape) == 3:
                    m = m[:, :, 0]  # Ensure the mask is 2D

                img = np.ones((m.shape[0], m.shape[1], 3))
                color_mask = c
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack((img, m * 0.3)))

                # Get the boundary of the mask
                contours = measure.find_contours(m, 0.5)
                for contour in contours:
                    contour = np.flip(contour, axis=1)
                    bound_polygons.append(Polygon(contour))
                    color.append(c)

                area = np.sum(m)
                if ann['category_id'] == 1:
                        blastomere_area.append(area)
                else:
                    fragment_total_area += area
                    # Calculate position in 3x3 grid
                    h, w = m.shape
                    grid_h, grid_w = h // 3, w // 3
                    for i in range(3):
                        for j in range(3):
                            if np.sum(m[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]) > 0:
                                cat2_positions[i, j] += 1
                if drawBbox:
                    # if ann['category_id'] != 2:
                    contours = maskUtils.toBbox([ann['segmentation']])
                    bbox_x, bbox_y, bbox_w, bbox_h = contours[0]
                    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y]]
                    np_poly = np.array(poly).reshape((4, 2))
                    bbox_polygon.append(Polygon(np_poly))
                    color.append(c)
        
        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.1)
        ax.add_collection(p)
        if paintBoundary:
            p = PatchCollection(bound_polygons, facecolor='none', edgecolors=color, linewidths=1)
            ax.add_collection(p)

    # Add grid lines for 3x3 grid
    ax = plt.gca()
    height, width = I.size
    total_area = height*width
    for i in range(1, 3):
        plt.axhline(y=i * height / 3, color='white', linestyle='--', linewidth=1)
        plt.axvline(x=i * width / 3, color='white', linestyle='--', linewidth=1)

    if drawDes:
        cat1_max_area = max(blastomere_area) if blastomere_area else 0
        cat1_min_area = min(blastomere_area) if blastomere_area else 0
        cat1_total_area = sum(blastomere_area) if blastomere_area else 0
        cat2_ratio = (fragment_total_area / cat1_total_area)*100 if cat1_total_area > 0 else 0
        cat2_dominant_position = np.unravel_index(np.argmax(cat2_positions), cat2_positions.shape)

        ax.text(10, 10, 
                f"Count of Blatomere: {count_cell}\n"
                f"Max Area of Blatomere: {cat1_max_area / 6400000:.4f}\n"
                f"Min Area of Blatomere: {cat1_min_area / 6400000:.4f}\n"
                f"Total Area Ratio (Fragment/Blatomere): {cat2_ratio:.2f}%\n"
                f"Dominant Position of Fragments: {cat2_dominant_position}\n", 
                fontsize=10, color='white',
                verticalalignment='top', horizontalalignment='left', 
                bbox=dict(facecolor='black', alpha=0.3))
    # plt.show()
    if isGT:
        img_path = img_path.split('/')[-1].split('.')[0]+'_gt.png'
    else:
        img_path = img_path.split('/')[-1].split('.')[0]+ '.png'
    plt.savefig('out/desc/'+img_path)
    plt.close()

def inference(cfg: Box,image_path):
    model = Model(cfg)
    model.setup()
    model.eval()
    # model.cuda()

    predictor = ModelPredictor(model)

    yolo = YOLO(cfg.yolo)

    yolo_res = yolo(image_path,conf=0.3,iou=0.83)[0]
    yolo_res = yolo_res[yolo_res.boxes.cls == 0]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bboxes = yolo_res.boxes.xyxy
    cls = [(cls + 1) for cls in yolo_res.boxes.cls.cpu().tolist()]
    score = yolo_res.boxes.conf.cpu().tolist()
    bboxes = torch.as_tensor(bboxes, device=model.sam.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(bboxes, image.shape[:2])
    predictor.set_image(image)
    masks, _, _,sem_mask = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    concat_mask = torch.cat((masks.squeeze(1) , sem_mask.squeeze(0)),dim=0)
    cls.append(2)
    score.append(1)
    res_anns = segmentationToCocoResult(concat_mask,cls,score,image_path)

    drawAnns(image_path,res_anns,False,True,True,False)
    return res_anns

def get_box(points):
    min_x = min_y = np.inf
    max_x = max_y = 0
    for x, y in points:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        # 'x,y,w,h 存储'
    return [min_x, min_y, max_x - min_x, max_y - min_y]

classname_to_id = {"cell":1,"fragment":2}
def annotation(ann_id,shape):
    label = shape['label']
    points = shape['points']
    annotation = {}
    annotation['id'] = ann_id
    annotation['image_id'] = 1
    label= label.split(" ")[0]
    annotation['category_id'] = int(classname_to_id[label])
    annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
    annotation['bbox'] = get_box(points)
    annotation['iscrowd'] = 0
    annotation['area'] = 1.0
    # print('annotation',annotation)
    return annotation

def showGT(image_path):
    anns_json = image_path.replace('jpg','json')
    ann_id = 0
    anns = []
    with open(anns_json,'r',encoding='utf-8') as f:
        obj = json.load(f)
        shapes = obj['shapes']
        for shape in shapes:
            ann = annotation(ann_id,shape)
            anns.append(ann)
            ann_id += 1
        drawAnns(image_path,anns,drawBbox=False,drawDes=True,paintBoundary=True,isGT=True)

if __name__ == "__main__":
    from config import cfg
    res_anns = []
    for id in ['197','198','199','200','201','281','282','283','284','285','286','297','298','300']:
    # for id in ['300']:
        image_path = 'testVideo/video/20230306-2559-6-F0-'+ id +'.jpg'
        hasJson=True
        res_anns += inference(cfg,image_path)
        if hasJson:
            showGT(image_path)
    with open('./video_res.json','w') as res:
        json.dump(res_anns,res,indent=1)