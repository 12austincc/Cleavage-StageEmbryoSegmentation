from box import Box

config = {
    "num_devices": 1,
    "batch_size": 1,
    "num_workers": 1,
    "num_epochs": 120,
    "num_classes": 2,
    "patience": 20,
    "eval_interval": 1,
    # "validate_cls": True,
    # "pretrained": True,
    "pretrained": False,
    "out_dir": "./out/training4",
    "visualize_dir": "sam_dual",
    "opt": {
        "learning_rate": 3e-4,
        "learning_rate_sam": 3e-5,
        "learning_rate_sem": 6e-4,
        "weight_decay": 1e-5,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_b',
        # "checkpoint": "../../sam_vit_h_4b8939.pth",  # 7955Mib
        # "checkpoint": "../../sam_vit_l_0b3195.pth",  # 6513Mib
        # "checkpoint": "../../sam_vit_b_01ec64.pth",  # 5319Mib
        "checkpoint": "./checkpoint/sam_vit_b.pth",
        "dual_checkpoint": "./out/training3/sam.pth",
        "freeze": {
            "sem_decoder": False,
            "image_encoder": False,
            "prompt_encoder": True,
            "mask_decoder": True,
        },
    },

    "dataset": {
        "train": {
            "root_dir": "./dataset/images/train",
            "annotation_file": "./dataset/annotations/instances_train.json"
        },
        "val": {
            "root_dir": "./dataset/images/val",
            "annotation_file": "./dataset/annotations/instances_val.json"
        },
    }
}

cfg = Box(config)
