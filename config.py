from box import Box

config = {
    "num_devices": 1,
    "batch_size": 2,
    "num_workers": 1,
    "num_epochs": 120,
    "num_classes": 2,
    "patience": 10,
    "eval_interval": 1,
    "pretrained": False,
    "out_dir": "./out/training",
    "visualize_dir": "sam_dual",
    "opt": {
        "learning_rate": 3e-4,
        "learning_rate_sam": 1e-4,
        "learning_rate_sem": 1.2e-3,
        "weight_decay": 4e-5,
        "decay_factor": 10,
        "steps": [4000, 9500],
        "warmup_steps": 150,
    },
    "model": {
        "type": 'vit_b',
        # "checkpoint": "../../sam_vit_h_4b8939.pth",  # 7955Mib
        # "checkpoint": "../../sam_vit_l_0b3195.pth",  # 6513Mib
        # "checkpoint": "../../sam_vit_b_01ec64.pth",  # 5319Mib
        "checkpoint": "./checkpoint/sam_vit_b.pth",
        "dual_checkpoint": "./checkpoint/sam_dual.pth",
        "freeze": {
            "sem_decoder": False,
            "image_encoder": False,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },

    "dataset": {
        "train": {
            "root_dir": "/images/train",
            "annotation_file": "/annotations/instances_train.json"
        },
        "val": {
            "root_dir": "/images/val",
            "annotation_file": "/annotations/instances_val.json"
        },
    },
    "yolo": './checkpoint/yolo.pt'
}

cfg = Box(config)
