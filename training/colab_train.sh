#!/bin/bash
set -euo pipefail
trap 'echo "ERROR: command failed on line $LINENO."; exit 1' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colab-friendly defaults. Override any of these before calling the script:
#   ARCH=yolov12 MODEL=yolov12n EPOCHS=50 BATCH_SIZE=16 SITE_INDICES=0,1 bash training/colab_train.sh
ARCH="${ARCH:-yolov8}"
MODEL="${MODEL:-}"
SLUG="${SLUG:-colab}"
SITE_INDICES="${SITE_INDICES:-0}"
IMG_SIZE="${IMG_SIZE:-640}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
YOLO_DEVICE="${YOLO_DEVICE:-0}"
FASTER_GPUS="${FASTER_GPUS:-1}"
FASTER_BATCH_SIZE="${FASTER_BATCH_SIZE:-2}"
AUTO_INSTALL="${AUTO_INSTALL:-1}"
AUTO_DOWNLOAD="${AUTO_DOWNLOAD:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"

STORAGE="${STORAGE:-/content}"
DATASET_DIR="${DATASET_DIR:-$STORAGE/dronewaste}"
TMP_DATASET_PATH="${TMP_DATASET_PATH:-$STORAGE/dronewaste_tmp}"
RESULTS_ROOT="${RESULTS_ROOT:-$STORAGE/kfold_results}"
RUN_ID="${RUN_ID:-${ARCH}_$(date +%Y-%m-%d)_${SLUG}}"

ZENODO_RECORD_ID="${ZENODO_RECORD_ID:-17045559}"
ZENODO_BASE_URL="https://zenodo.org/records/${ZENODO_RECORD_ID}/files"
DATASET_JSON="dronewaste_v1.0.json"
INFO_FILE="info.txt"
IMAGES_ARCHIVE="images.tar.gz"

export WANDB_MODE

case "$ARCH" in
    yolov8)
        MODEL="${MODEL:-yolov8n}"
        ;;
    yolov12)
        MODEL="${MODEL:-yolov12n}"
        ;;
    faster)
        MODEL="${MODEL:-faster-rcnn_r50_fpn_mstrain_3x_coco}"
        ;;
    *)
        echo "Invalid ARCH=$ARCH. Use yolov8, yolov12, or faster."
        exit 1
        ;;
esac

download_file() {
    local url="$1"
    local target="$2"
    local expected_md5="$3"

    if [ -f "$target" ] && echo "${expected_md5}  ${target}" | md5sum -c --status; then
        echo "Found $(basename "$target") with matching checksum."
        return
    fi

    if [ -f "$target" ]; then
        mv "$target" "${target}.bad.$(date +%s)"
    fi

    echo "Downloading $(basename "$target") ..."
    curl -L --retry 5 --retry-delay 10 -C - -o "$target" "$url"
    echo "${expected_md5}  ${target}" | md5sum -c -
}

download_dataset() {
    mkdir -p "$DATASET_DIR"

    download_file \
        "${ZENODO_BASE_URL}/${DATASET_JSON}?download=1" \
        "$DATASET_DIR/$DATASET_JSON" \
        "87469b3641aade7c4580b8f0ed5c6300"

    download_file \
        "${ZENODO_BASE_URL}/${INFO_FILE}?download=1" \
        "$DATASET_DIR/$INFO_FILE" \
        "0402a72a4eb18cdb97a58fec0d8dfda2"

    local image_count=0
    if [ -d "$DATASET_DIR/images" ]; then
        image_count="$(find "$DATASET_DIR/images" -maxdepth 1 -type f | wc -l)"
    fi

    if [ "$image_count" -lt 4993 ]; then
        download_file \
            "${ZENODO_BASE_URL}/${IMAGES_ARCHIVE}?download=1" \
            "$DATASET_DIR/$IMAGES_ARCHIVE" \
            "0e2c2b2424737b94e9cfa28ff90e6544"

        echo "Extracting images archive ..."
        tar -xzf "$DATASET_DIR/$IMAGES_ARCHIVE" -C "$DATASET_DIR"
    else
        echo "Found $image_count images in $DATASET_DIR/images."
    fi
}

install_common_deps() {
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r "$REPO_ROOT/requirements.txt"
}

install_yolov8_deps() {
    install_common_deps
    python -m pip install ultralytics
}

install_yolov12_deps() {
    install_common_deps

    local yolov12_dir="${YOLOV12_DIR:-/content/yolov12}"
    local filtered_requirements
    if [ ! -d "$yolov12_dir/.git" ]; then
        git clone https://github.com/sunsmarterjie/yolov12 "$yolov12_dir"
    fi

    filtered_requirements="$(mktemp)"
    grep -Eiv 'flash[-_]?attn|flash_attn|\.whl|^onnx([<=> ].*)?$|^onnxruntime([<=> ].*)?$' "$yolov12_dir/requirements.txt" > "$filtered_requirements"
    python -m pip install -r "$filtered_requirements"
    python -m pip install "onnx>=1.17.0" "onnxruntime>=1.17.0"
    rm -f "$filtered_requirements"

    if [ "${INSTALL_FLASH_ATTN:-0}" = "1" ]; then
        python -m pip install flash-attn --no-build-isolation
    fi

    python -m pip install -e "$yolov12_dir" --no-deps
}

install_faster_deps() {
    install_common_deps
    python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
    python -m pip install -U openmim
    python -m pip install mmengine
    python -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
    python -m pip install mmdet
}

install_deps() {
    case "$ARCH" in
        yolov8)
            install_yolov8_deps
            ;;
        yolov12)
            install_yolov12_deps
            ;;
        faster)
            install_faster_deps
            ;;
    esac
}

if [ "$AUTO_INSTALL" = "1" ]; then
    install_deps
fi

if [ "$AUTO_DOWNLOAD" = "1" ]; then
    download_dataset
fi

mkdir -p "$TMP_DATASET_PATH" "$RESULTS_ROOT/$RUN_ID" "$SCRIPT_DIR/logs"

echo ""
echo "========================================="
echo "DroneWaste Colab training"
echo "Arch: $ARCH"
echo "Model: $MODEL"
echo "Sites: $SITE_INDICES"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Dataset: $DATASET_DIR/$DATASET_JSON"
echo "Results: $RESULTS_ROOT/$RUN_ID"
echo "W&B mode: $WANDB_MODE"
echo "========================================="
echo ""

cd "$SCRIPT_DIR"
IFS=',' read -r -a SITE_ARRAY <<< "$SITE_INDICES"

for SITE_INDEX in "${SITE_ARRAY[@]}"; do
    FOLD_ID="${SITE_INDEX}_$(date +%s)"

    echo ""
    echo "Running fold $FOLD_ID for site index $SITE_INDEX"
    echo ""

    python kfold_train.py \
        --arch "$ARCH" \
        --site_index "$SITE_INDEX" \
        --run_id "$FOLD_ID" \
        --dataset_path "$DATASET_DIR/$DATASET_JSON" \
        --tmp_dataset_path "$TMP_DATASET_PATH" \
        --results_path "$RESULTS_ROOT/$RUN_ID" \
        --model "$MODEL" \
        --img_size "$IMG_SIZE" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --yolo_batch_size "$BATCH_SIZE" \
        --yolo_device "$YOLO_DEVICE" \
        --faster_batch_size "$FASTER_BATCH_SIZE" \
        --faster_gpus "$FASTER_GPUS" \
        2>&1 | tee "$SCRIPT_DIR/logs/log_colab_${FOLD_ID}.out"
done

echo ""
echo "Training completed. Results saved to $RESULTS_ROOT/$RUN_ID"
