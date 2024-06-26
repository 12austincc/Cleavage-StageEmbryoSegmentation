import numpy as np
from pycocotools import mask
import torch
# import base64

def segmentationToCocoResult(labelMap, cls, score, imgId):
    '''
    Convert a segmentation map to COCO stuff segmentation result format.
    :param labelMap: [n x h x w] segmentation maps of every instance
    :param imgId: the id of the COCO image (last part of the file name)
    :return: anns    - a list of dicts for each label in this image
       .image_id     - the id of the COCO image
       .category_id  - the id of the stuff class of this annotation
       .segmentation - the RLE encoded segmentation of this class
       .score        - the conf of this box
    '''

    # Get stuff labels
    labelsStuff = torch.Tensor(cls)
    labelMap = labelMap.cpu()
    # Add stuff annotations
    anns = []
    for i in range(0,labelsStuff.shape[0]):

        labelId = labelsStuff[i]
        # Create mask and encode it
        Rs = segmentationToCocoMask(labelMap[i])

        # Create annotation data and add it to the list
        anndata = {}
        anndata['image_id'] = int(imgId)
        anndata['category_id'] = int(labelId)
        anndata['segmentation'] = {
            'size': Rs['size'],
            # 'counts': base64.b64encode(Rs['counts']).decode('utf-8')
            'counts': Rs['counts'].decode('utf-8')
        }
        # anndata['segmentation'] = Rs
        anndata['score'] = float(score[i])
        anns.append(anndata)
    return anns


def segmentationToCocoMask(labelMap):
    '''
    Encodes a segmentation mask using the Mask API.
    :param labelMap: [h x w] binary segmentation mask
    :return: Rs - the encoded label mask for label 'labelId'
    '''
    labelMask = labelMap
    labelMask = np.expand_dims(labelMask, axis=2)
    labelMask = labelMask.astype('uint8')
    labelMask = np.asfortranarray(labelMask)
    Rs = mask.encode(labelMask)
    assert len(Rs) == 1
    Rs = Rs[0]

    return Rs

def objectToCocoResult(bbox,cls,score,imgId):
    # print('save object detection result in coco format')

    anns = []
    for i in range(0,cls.shape[0]):

        labelId = cls[i]

        # Create annotation data and add it to the list
        anndata = {}
        anndata['image_id'] = int(imgId)
        anndata['category_id'] = int(labelId)
        anndata['bbox'] = bbox[i].tolist()
        anndata['score'] = float(score[i])
        anns.append(anndata)
    return anns


if __name__ == '__main__':
    import torch
    import json
    image_id = 0
    masks = torch.randint(0,1,(7,800,800))
    score = torch.rand(3,)
    cls = torch.randint(0,3,(3,))
    masks = torch.tensor([
        [[0, 0, 1, 1, 0],
         [0, 1, 1, 1, 0],
         [1, 1, 0, 0, 0],
         [0, 1, 1, 1, 1],
         [0, 0, 1, 0, 0]],

        [[1, 1, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [0, 0, 0, 1, 1],
         [0, 0, 0, 1, 1],
         [0, 0, 1, 1, 1]],

        [[0, 0, 0, 1, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 0, 0, 0],
         [1, 1, 1, 0, 0],
         [0, 0, 0, 1, 1]]
    ])
    # for i in range(3):
    #     masks[i] = torch.rand(800,800)

    # anns = segmentationToCocoResult(masks,cls, score,image_id)
    bbox = torch.rand(5,4)
    anns = objectToCocoResult(bbox,cls,score,image_id)
    print(anns)
    '''
    '''
    # with open('anns.json','w') as anns_json:
    #     json.dump(anns,anns_json,indent=1)