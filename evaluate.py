from utils import COCOeval
from utils import evalSemantic

if __name__ == '__main__':
    from pycocotools.coco import COCO
    path  = './dataset/annotations/instances_val.json'
    cocoGT = COCO(path)
    res_json= './res/res_sam_dual.json'
    evalSemantic(cocoGT,res_json)
    evalIns = True
    if evalIns:
        models = ['cascade_maskrcnn','donet','mask2former','maskdino','ms_rcnn','rtmdet','solov2','yolact']
        models = ['sam_dual']
        for model in models:
            cocoDT = cocoGT.loadRes('/res/'+model+'_res.json')
            print(model)
            wrongIds = []
            imgIds = [id for id in cocoGT.getImgIds() if id not in wrongIds]
            annType = ['segm','bbox']
            cocoEval = COCOeval(cocoGT,cocoDT,annType[0])
            cocoEval.params.imgIds = imgIds
            cocoEval.params.catIds = [1]
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            # 打印新增的统计指标
            print('catId:',cocoEval.params.catIds)
            # print('Dice: {:.3f}'.format(cocoEval.stats[12])) #Dice=F1
            print('F1: {:.3f}'.format(cocoEval.stats[13]))
            print('Jaccard: {:.3f}'.format(cocoEval.stats[14]))
            print('Precision: {:.3f}'.format(cocoEval.stats[15]))
            print('Recall: {:.3f}'.format(cocoEval.stats[16]))
            print('Accuracy: {:.3f}'.format(cocoEval.stats[17]))