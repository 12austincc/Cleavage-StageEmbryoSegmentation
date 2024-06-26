import torch.nn as nn
import torch.nn.functional as F
import torch
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from semantic_decoder import SemanticDecoder

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self):
        self.sam = sam_model_registry[self.cfg.model.type](checkpoint=self.cfg.model.checkpoint)
        self.semantic_decoder = SemanticDecoder(num_classes=self.cfg.num_classes)
        if self.cfg.pretrained:
            state_dict = torch.load(self.cfg.model.dual_checkpoint)
            self.sam.load_state_dict(state_dict[0])
            self.semantic_decoder.load_state_dict(state_dict[1])
        self.sam.train()
        self.semantic_decoder.train()
        if self.cfg.model.freeze.image_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.prompt_encoder:
            for param in self.sam.prompt_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.mask_decoder:
            for param in self.sam.mask_decoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.sem_decoder:
            for param in self.semantic_decoder.parameters():
                param.requires_grad = False
        

    def forward(self, images, bboxes):
        _, _, H, W = images.shape
        image_embeddings = self.sam.image_encoder(images)
         # image_embeddings : (1,256,64,64)
        sem_mask = self.semantic_decoder(image_embeddings)
        
        ins_masks = []
        ious = []
        for embedding, bbox in zip(image_embeddings, bboxes):
            # bbox (N,4)
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None,
                boxes=bbox,
                masks=None,
            )

            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            ins_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)


        return sem_mask,ins_masks,ious