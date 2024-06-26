from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import os
from PIL import Image
from pycocotools import mask as maskUtils

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
from skimage import measure
from shapely.geometry import Polygon as shapelyPolygon


def showAnns(coco:COCO,dataset_dir,subset,imgIds,catIds,model,drawBbox=False,drawDes=False,drawLabel=False,save=False,show=False,paintBoundary=False):
    imgs = coco.loadImgs(imgIds)
    cat_names = {cat['id']: cat['name'] for cat in coco.cats.values()}

    images_per_row = 7
    num_images = len(imgIds)
    num_rows = (num_images + images_per_row - 1) // images_per_row  # Calculate the number of rows needed

    fig, axs = plt.subplots(num_rows, images_per_row, figsize=(20, num_rows * 4))
    axs = axs.flatten()  # Flatten the 2D array of axes for easy iteration

    for ax in axs:
        ax.axis('off')

    for i,img in enumerate(imgs):
        annIds = coco.getAnnIds(img['id'],catIds)
        anns = coco.loadAnns(annIds)
        I =Image.open(os.path.join(dataset_dir,"images",subset,img['file_name'])) 
        ax = axs[i]
        ax.imshow(I)
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        count_cell = 0
        total_area = 0
        area_cat_fragment = 0
        for ann in anns:
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
                        color.append(c)
                        poly_shapely = shapelyPolygon(poly)
                        total_area += poly_shapely.area
                        if ann['category_id'] == 2:
                            area_cat_fragment += poly_shapely.area
                    if drawBbox:
                        [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y]]
                        np_poly = np.array(poly).reshape((4, 2))
                        polygons.append(Polygon(np_poly))
                        color.append(c)
                        if drawLabel:
                            cat_id = ann['category_id']
                            cat_name = cat_names[cat_id]
                            plt.text(bbox_x, bbox_y, cat_name, fontsize=12, color=c,
                             verticalalignment='top', horizontalalignment='left',
                            bbox=dict(facecolor='white', alpha=0.7, pad=0))
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
                    if ann['iscrowd'] == 1:
                        color_mask = np.array([2.0, 166.0, 101.0]) / 255
                    if ann['iscrowd'] == 0:
                        # color_mask = np.random.random((1, 3)).tolist()[0]
                        color_mask = c
                    for i in range(3):
                        img[:, :, i] = color_mask[i]
                    ax.imshow(np.dstack((img, m * 0.1)))

                    # Get the boundary of the mask
                    contours = measure.find_contours(m, 0.5)
                    for contour in contours:
                        contour = np.flip(contour, axis=1)
                        polygons.append(Polygon(contour))
                        color.append(c)
                        contour_shapely = shapelyPolygon(contour)
                        total_area += contour_shapely.area
                        if ann['category_id'] == 2:
                            area_cat_fragment += contour_shapely.area


                    if drawBbox:
                        if ann['category_id'] != 2:
                            contours = maskUtils.toBbox([ann['segmentation']])
                            bbox_x, bbox_y, bbox_w, bbox_h = contours[0]
                            poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y]]
                            np_poly = np.array(poly).reshape((4, 2))
                            polygons.append(Polygon(np_poly))
                            color.append(c)
                            if drawLabel:
                                mid_x = bbox_x + bbox_w / 2
                                mid_y = bbox_y + bbox_h / 2
                                category = coco.loadCats(ann['category_id'])[0]['name']
                                ax.text(mid_x, mid_y, category, fontsize=12, color=c, verticalalignment='bottom', horizontalalignment='right')
        
        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        if paintBoundary:
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)

        if total_area > 0:
            area_ratio_cat_fragment = area_cat_fragment / total_area
        else:
            area_ratio_cat_fragment = 0
        if drawDes:
            # ax.text(10, 10, f"Count of Blatomere: {count_cell}\nFragment Ratio: {area_ratio_cat_fragment:.2f}", fontsize=12, color='white',
            #      verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='black', alpha=0.5))
            ax.text(10, 10, f"Count of Blatomere: {count_cell}\n", fontsize=12, color='white',
                 verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='black', alpha=0.5))
    
    for ax in axs[num_images:]:
        ax.axis('off')  # Turn off axis for empty subplots

    plt.tight_layout()
    if save:
        plt.savefig('/Users/ios/Desktop/论文/囊胚预测/detect/cell/'+model+'.png')
    if show:
        plt.show()

def showBbox(dataset_dir,subset,imgIds,catIds,drawDes=False,drawLabel=False):
    cocoGT = COCO('/Users/ios/Downloads/已标记图片/dataset/datasetcoco/annotations/instances_val.json')
    coco = cocoGT.loadRes('/Users/ios/Downloads/res/yolov8_res.json')
    # coco = cocoGT
    imgs = coco.loadImgs(imgIds)
    cat_names = {cat['id']: cat['name'] for cat in coco.cats.values()}

    images_per_row = 7
    num_images = len(imgIds)
    num_rows = (num_images + images_per_row - 1) // images_per_row  # Calculate the number of rows needed

    fig, axs = plt.subplots(num_rows, images_per_row, figsize=(20, num_rows * 4))
    axs = axs.flatten()  # Flatten the 2D array of axes for easy iteration

    for ax in axs:
        ax.axis('off')

    for i,img in enumerate(imgs):
        annIds = coco.getAnnIds(img['id'],catIds)
        anns = coco.loadAnns(annIds)
        I =Image.open(os.path.join(dataset_dir,"images",subset,img['file_name']))
        I = I.crop((0,0,800,767))
        ax = axs[i]
        ax.imshow(I)
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        count_cell = 0
        for ann in anns:
            # c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            if ann['category_id'] == 1:
                c = [1,0,0]
                count_cell += 1
            elif ann['category_id'] == 2:
                c = [0,1,0]
            if 'score' in ann:
                if ann['score'] <=0.3:
                    if ann['category_id'] == 1:
                        count_cell -=1
                    continue

                [bbox_x1, bbox_y1, bbox_x2, bbox_y2] = ann['bbox']
            else:
                [bbox_x1, bbox_y1, bbox_w, bbox_h] = ann['bbox']
                bbox_x2 = bbox_x1 + bbox_w
                bbox_y2 = bbox_y1 + bbox_h
            poly = [[bbox_x1, bbox_y1], [bbox_x1, bbox_y2], [bbox_x2, bbox_y2], [bbox_x2 , bbox_y1]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons.append(Polygon(np_poly))
            color.append(c)

        # p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
        # ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=1)
        ax.add_collection(p)

        ax.text(10, 10, f"Count of Blatomere: {count_cell}\n", fontsize=8, color='white',
                 verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='black', alpha=0.5))
    
    for ax in axs[num_images:]:
        ax.axis('off')  # Turn off axis for empty subplots

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset_dir = './dataset/datasetcoco'
    subset='val'
    cocoGT = COCO("{}/annotations/instances_{}.json".format(dataset_dir, subset))
    imgId = [2,84,57,64,69,96,108]
    catId = [2]
    print(imgId,catId)
    # for gt:
    # coco = cocoGT 
    # showAnns(coco,dataset_dir,subset,imgId,catId,'gt',drawDes=False,save=True,show=True,paintBoundary=False)

    # for instance
    # models = ['cascade_maskrcnn','donet','mask2former','maskdino','ms_rcnn','rtmdet','solov2','yolact']
    # for model in models:
    #     coco = cocoGT.loadRes('/Users/ios/Downloads/res/instance segmentation/'+model+'_res.json')
    #     showAnns(coco,dataset_dir,subset,imgId,catId,model,drawDes=True)

    # for semantic
    # models = ['unet','unetpp','transunet','deeplabv3','segformer','yolosam']
    # for model in models:
    #     coco = cocoGT.loadRes('/Users/ios/Downloads/res/semantic segmentation/'+model+'_res.json')
    #     showAnns(coco,dataset_dir,subset,imgId,catId,model,drawDes=False,save=True,show=False,paintBoundary=False)

    # for object detection
    # showBbox(dataset_dir,subset,imgId,catId)

 