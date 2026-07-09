import os
import shutil
import random
import numpy as np
import yaml
import torch
from ultralytics import YOLO
import time
from datetime import datetime
import argparse
import wandb

# reference: network architecture
# https://github.com/sunsmarterjie/yolov12

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--dataset_yaml", type=str, required=True)
parser.add_argument("--results_path", type=str, required=True)
parser.add_argument("--img_size", type=int, default=640)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--patience", type=int, default=50)
parser.add_argument("--lr_tl", type=float, default=0.001)  # 1e-3
parser.add_argument("--lr_ft", type=float, default=0.0001)  # 1e-4
parser.add_argument("--device", type=str, required=True)
parser.add_argument("--fold_id", type=str, default=None)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--cache", type=str, default="disk")
parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

# CLI arguments
MODEL = args.model
DATASET = args.dataset_yaml
RESULTS_PATH = args.results_path
IMG_SIZE = args.img_size
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
PATIENCE = args.patience
LR_TL = args.lr_tl
LR_FT = args.lr_ft
DEVICE = args.device
WORKERS = args.workers
DETERMINISTIC = args.deterministic
RUN = datetime.now().strftime("%Y-%m-%d_%H-%M")  # get current timestamp


def parse_cache(value):
    value = value.strip()
    if value.lower() in ("false", "0", "no", "off", "none"):
        return False
    if value.lower() in ("true", "1", "yes", "on"):
        return True
    return value


# training parameters
optimizer = "AdamW"
frozen_layers_tl = 10  # freeze backbone layers
frozen_layers_ft = 2  # freeze backbone layers during fine tuning
train_cache = parse_cache(args.cache)  # disk cache required for deterministic training
val_during_train = True  # enable validation during training
train_save_plots = True  # save plots of training and validation metrics

# evaluation parameters
object_conf_thres = 0.001
iou_thres = 0.5
eval_save_json = True
eval_save_plots = True

# augmentation parameters
degrees = 90.0
translate = 0.2
scale = 0.1
flipud = 0.5
fliplr = 0.5
mosaic = 1.0  # combines four training images into one (default=1.0, set 0.0 to disable)
mixup = 1.0  # mixup coefficient (default=0.0, set 0.0 to disable)

hsv_h=0
hsv_s=0
hsv_v=0


def set_deterministic(seed=1337, deterministic=True):
    # Python random module
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Additional PyTorch deterministic settings
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def initialize_wandb():
    mode = os.environ.get("WANDB_MODE", "offline")
    api_key = os.environ.get("WANDB_API_KEY")

    if api_key and mode != "offline":
        wandb.login(key=api_key)
    elif mode != "offline":
        print("WANDB_API_KEY is not set; using offline W&B logging.")
        mode = "offline"
        os.environ["WANDB_MODE"] = mode

    wandb.init(mode=mode, project=wandb_project)


def run_dir_candidates(run_name):
    project_leaf = os.path.basename(os.path.normpath(wandb_project))
    candidates = [
        os.path.join(wandb_project, run_name),
        os.path.join("runs", "detect", wandb_project, run_name),
        os.path.join("runs", "detect", project_leaf, run_name),
    ]
    return list(dict.fromkeys(candidates))


def resolve_run_dir(run_name):
    candidates = run_dir_candidates(run_name)
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find Ultralytics run folder for {run_name}. "
        f"Checked: {', '.join(candidates)}"
    )


def resolve_best_weights(run_name):
    candidates = [
        os.path.join(run_dir, "weights", "best.pt")
        for run_dir in run_dir_candidates(run_name)
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find best.pt for {run_name}. Checked: {', '.join(candidates)}"
    )


def move_run_artifacts(run_name, dst):
    shutil.move(src=resolve_run_dir(run_name), dst=dst)


def transfer_learning():
    print()
    print(f"[{RUN}] transfer learning ...")
    print()

    model = YOLO(f"{MODEL}.pt")

    model.train(
        data=data_path,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=wandb_project,
        name=project_name + "_tl",  # create a run folder
        freeze=frozen_layers_tl,  # freeze layers
        epochs=EPOCHS,
        patience=PATIENCE,
        optimizer=optimizer,
        verbose=False,
        seed=seed,
        deterministic=DETERMINISTIC,
        workers=WORKERS,
        amp=False,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        flipud=flipud,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        lr0=LR_TL,
        cache=train_cache,
        val=val_during_train,
        plots=train_save_plots,  # plots saving is required to avoid errors  
    )
    # release GPU and free RAM by deleting the model
    del model
    torch.cuda.empty_cache()


def fine_tuning():
    print()
    print(f"[{RUN}] fine tuning ...")
    print()

    model = YOLO(resolve_best_weights(project_name + "_tl"))

    model.train(
        data=data_path,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=wandb_project,
        name=project_name + "_ft",  # create a run folder
        freeze=frozen_layers_ft,  # freeze layers
        epochs=EPOCHS,
        patience=PATIENCE,
        optimizer=optimizer,
        verbose=False,
        seed=seed,
        deterministic=DETERMINISTIC,
        workers=WORKERS,
        amp=False,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        flipud=flipud,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        lr0=LR_FT,
        cache=train_cache,
        val=val_during_train,
        plots=train_save_plots,  # plots saving is required to avoid errors
    )
    del model
    torch.cuda.empty_cache()
    return


def evaluate_model(model, split):
    print()
    print(f'[{RUN}] evaluating model on "{split}" set ...')
    print()

    model = YOLO(model)

    metrics = model.val(
        project=wandb_project,  # save inside the run folder
        name=f'{project_name}_{split}',  # create a split folder
        split='val' if split == 'valid' else 'test',
        conf=object_conf_thres,
        iou=iou_thres,
        save_json=eval_save_json,
        plots=eval_save_plots,
    )

    # reference: validation metrics
    # https://docs.ultralytics.com/reference/utils/metrics/#ultralytics.utils.metrics.Metric
    box = metrics.box
    return {
        "p": box.mp,  # mean precision
        "r": box.mr,  # mean recall
        "map50": box.map50,  # mean AP @ 0.5
        "map75": box.map75,  # mean AP @ 0.75
        "map": box.map,  # mean AP @ 0.5-0.95
    }


def complete_training(output_path, project_name, val, test):
    print()
    print(f"[{RUN}] saving results ...")
    print()

    # create project folder
    os.makedirs(os.path.join(output_path, project_name), exist_ok=True)

    # save metrics
    output = f"{RUN};"
    output += f'{val["p"]:.4f};{val["r"]:.4f};{val["map50"]:.4f};{val["map75"]:.4f};{val["map"]:.4f};'
    output += f'{test["p"]:.4f};{test["r"]:.4f};{test["map50"]:.4f};{test["map75"]:.4f};{test["map"]:.4f}\n'
    with open(os.path.join(output_path, "runs.txt"), "a") as f:
        f.write(output.replace(".", ","))

    # move artifacts to results folder
    move_run_artifacts(project_name + "_tl", os.path.join(output_path, project_name, "tl"))
    move_run_artifacts(project_name + "_ft", os.path.join(output_path, project_name, "ft"))
    move_run_artifacts(project_name + "_valid", os.path.join(output_path, project_name, "valid"))
    move_run_artifacts(project_name + "_test", os.path.join(output_path, project_name, "test"))
    shutil.move(
        src=os.path.join(wandb_project, f"{project_name}_params.yaml"),
        dst=os.path.join(output_path, project_name, "params.yaml"),
    )


def initialize_project(project_name):
    print()
    print(f'[{RUN}] initializing project "{project_name}" ...')
    print()

    os.makedirs(wandb_project, exist_ok=True)

    # save kfold parameters
    with open(os.path.join(wandb_project, f"{project_name}_params.yaml"), "w") as f:
        yaml.safe_dump(
            {
                "model": MODEL,
                "dataset": DATASET,
                "img_size": IMG_SIZE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "patience": PATIENCE,
                "lr_tl": LR_TL,
                "lr_ft": LR_FT,
                "optimizer": optimizer,
                "workers": WORKERS,
                "deterministic": DETERMINISTIC,
                "train_cache": train_cache,
                "frozen_layers_tl": frozen_layers_tl,
                "frozen_layers_ft": frozen_layers_ft,
                "object_conf_thres": object_conf_thres,
                "iou_thres": iou_thres,
                "degrees": degrees,
                "translate": translate,
                "scale": scale,
                "flipud": flipud,
                "fliplr": fliplr,
                "mosaic": mosaic,
                "mixup": mixup,
                "hsv_h": hsv_h,
                "hsv_s": hsv_s,
                "hsv_v": hsv_v,
                "seed": seed,
            },
            f,
            sort_keys=False,
        )


seed = 1337
set_deterministic(seed=seed, deterministic=DETERMINISTIC)

# during kfold training, use the fold id as the project name
project_name = args.fold_id

# define project details and paths
wandb_project = "uav_waste"
data_path = DATASET

# initialize wandb
initialize_wandb()

# create project folder and save parameters
initialize_project(project_name)

# train: transfer learning
tl_start = time.time()
transfer_learning()
tl_end = time.time()

BATCH_SIZE = BATCH_SIZE // 2  # reduce batch size for fine tuning

# train: fine tuning
ft_start = time.time()
fine_tuning()
ft_end = time.time()

# evaluate model
model = resolve_best_weights(project_name + "_ft")
val_metrics = evaluate_model(model, "valid")
test_metrics = evaluate_model(model, "test")

# save fold results and move artifacts
complete_training(RESULTS_PATH, project_name, val_metrics, test_metrics)

print()
print(f'transfer learning: {(tl_end-tl_start)/60:.3f} minutes')
print(f'fine tuning: {(ft_end-ft_start)/60:.3f} minutes')
print(f"[{RUN}] done!")
