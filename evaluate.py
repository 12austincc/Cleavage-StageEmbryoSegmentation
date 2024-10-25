from utils import COCOeval
from utils import evalSemantic

if __name__ == '__main__':
    from pycocotools.coco import COCO
    ins_res  = './res/instances_val.json'
    sem_res = './res/fragment_res5.json'
    cocoGT = COCO(ins_res)
    cocoSem = COCO(sem_res)
    res_json= './res/sam_dual_res.json'
    # evalSemantic(cocoSem,res_json)
    evalIns = True
    if evalIns:
        models = ['sam_dual']
        for model in models:
            cocoDT = cocoGT.loadRes(res_json)
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
            print('F1: {:.3f}'.format(cocoEval.stats[12]))
            print('Precision: {:.3f}'.format(cocoEval.stats[13]))
            print('Recall: {:.3f}'.format(cocoEval.stats[14]))