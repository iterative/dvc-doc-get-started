#!/bin/bash

set -veux

export REPO_PATH="${REPO_ROOT}"/pipelines

# Set the following to fashion-${DATASET} to change the project dataset 

mkdir -p "$REPO_PATH"
pushd "${REPO_PATH}"

virtualenv -p python3 .venv
export VIRTUAL_ENV_DISABLE_PROMPT=true
source .venv/bin/activate
echo '.venv/' > .gitignore
pip install 'dvc[all]'

git init
git checkout -b main
cp $HERE/code-pipelines/README.md .

tag_tick
git add .
git commit -m  "Initialize Git repository"
git tag -a "git-init" -m "Git initialized."


tag_tick
dvc init
git commit -m "Initialize DVC project"
git tag -a "dvc-init" -m "DVC initialized."

# Remote active on this env only, for writing to HTTP redirect below.
dvc remote add -d --local storage s3://dvc-public/remote/get-started
# Actual remote for generated project (read-only). Redirect of S3 bucket above.
dvc remote add -d storage https://remote.dvc.org/get-started

tag_tick
git add .
git commit -m "Configure default remote"
git tag -a "config-remote" -m "Read-only remote storage configured."

test -d data/mnist || mkdir -p data/mnist

dvc import https://github.com/iterative/dataset-registry \
           mnist/raw -o data/mnist/raw

tag_tick
git add data/mnist/raw.dvc data/mnist/.gitignore
git commit -m "Add raw MNIST data"
git tag -a "import-mnist-data" -m "MNIST data file added."
dvc push

cp -r "${HERE}"/code-pipelines/src .
cp "${HERE}"/code-pipelines/requirements.txt .
cp "${HERE}"/code-pipelines/params.yaml .
pip install -r "${REPO_PATH}"/requirements.txt

tag_tick
git add .
git commit -m "Add source code files to repo"
git tag -a "source-code" -m "Source code added."

dvc stage add -n prepare \
              -p prepare.seed \
              -p prepare.remix \
              -p prepare.remix_split \
              -d data/mnist/raw \
              -d src/prepare.py \
              -o data/mnist/prepared \
              python3 src/prepare.py

dvc repro prepare 

tag_tick
git add data/mnist/.gitignore dvc.yaml dvc.lock
git commit -m "Create data preparation stage"
dvc push
git tag -a "prepare" -m "First pipeline stage (data preparation) created."

dvc stage add -n preprocess \
    -p preprocess.seed \
    -p preprocess.normalize \
    -p preprocess.shuffle \
    -p preprocess.add_noise \
    -p preprocess.noise_amount \
    -p preprocess.noise_s_vs_p \
    -d data/mnist/prepared/ \
    -d src/preprocess.py \
    -o data/mnist/preprocessed/ \
    python3 src/preprocess.py

dvc repro preprocess
dvc push
tag_tick
git add data/mnist/.gitignore dvc.yaml dvc.lock
git commit -m "Second pipeline stage (preprocessing) created"
git tag -a "preprocess" -m "Second pipeline stage (data preprocessing) created."

mkdir -p models/mnist
dvc stage add -n train \
              -p train.seed \
              -p train.validation_split \
              -p train.epochs \
              -p train.batch_size \
              -p model.name \
              -p model.optimizer \
              -p model.mlp.units \
              -p model.mlp.activation \
              -p model.cnn.dense_units \
              -p model.cnn.conv_kernel_size \
              -p model.cnn.conv_units \
              -p model.cnn.dropout \
              -d src/train.py \
              -d src/models.py \
              -d data/mnist/preprocessed/ \
              -o models/mnist/model.h5 \
              --plots-no-cache logs.csv \
              python3 src/train.py

tag_tick
dvc repro train
git add .gitignore dvc.yaml dvc.lock
git status  -s
git commit -m "Created training stage"
dvc push
git tag -a "training" -m "Training stage created."

dvc stage add -n evaluate \
              -d src/evaluate.py \
              -d models/mnist/model.h5 \
              -M metrics.json \
              python3 src/evaluate.py
dvc repro evaluate

tag_tick
git add .gitignore dvc.yaml dvc.lock metrics.json
git commit -m "Create evaluation stage"
dvc push
git tag -a "evaluation" -m "Evaluation stage created."

