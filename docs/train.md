# Training instructions

## Parameters definition

Model training is performed using the [training script](../training/train.sh).

At the beginning of the script, some parameters must be defined:

* `arch`: the model architecture to be trained (`yolov8`, `yolov12`, `faster`)
* `slug`: a unique identifier for the run (optional, default: `train`)
* `$STORAGE` env variable: path to the storage folder for the dataset and results
* `$TMPDIR` env variable: path to temporary dataset generated for each fold
* `$ENVSDIR` env variable: path to the virtual environments folder

## Dataset download

Download the DroneWaste dataset from [Zenodo](https://doi.org/10.5281/zenodo.17045559) and extract it in the `$STORAGE/dronewaste` folder.
To extract the `images/` folder, use the the following command:

```bash
tar xzf images.tar.gz
```

The structure of the dataset should be as follows:

```bash
$STORAGE/dronewaste/
├── dronewaste_v1.0.json
├── info.txt
└── images/
    ├── site_1_4.png
    ├── site_1_5.png
    └── ...
```

## Training script

Start the model training procedure using the following command:

```bash
cd training/
bash train.sh
```

The training script will select the correct environment based on the selected model architecture.

Model training on the DroneWaste dataset is performed using a k-fold cross-validation approach where each site is treated as a separate fold.
The training script iterates over all sites. At each iteration, a temporary dataset is generated where the current site is used as the test set while the remaining sites are used as the training set.

The results from each fold training are saved in separate folders inside the `$STORAGE/kfold_results/$run_id` folder.

## Google Colab

Use the Colab helper script to install dependencies, download the DroneWaste dataset from Zenodo, and start training automatically:

```bash
cd /content/dronewaste
bash training/colab_train.sh
```

The default Colab run uses `yolov8n`, one site fold, 10 epochs, single GPU, and offline W&B logging so that it can start on a standard Colab runtime. Override the settings with environment variables:

```bash
ARCH=yolov12 MODEL=yolov12n EPOCHS=50 BATCH_SIZE=16 SITE_INDICES=0,1 \
  bash training/colab_train.sh
```

For full cross-validation, set `SITE_INDICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16`. Results are written to `/content/kfold_results` by default. Set `STORAGE=/content/drive/MyDrive` if you want dataset and results to persist in Google Drive.
