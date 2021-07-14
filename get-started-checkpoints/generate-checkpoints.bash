#!/bin/bash

set -veux

export REPO_PATH="${REPO_ROOT}"/checkpoints

mkdir -p "$REPO_PATH"
pushd "${REPO_PATH}"

virtualenv -p python3 .venv
export VIRTUAL_ENV_DISABLE_PROMPT=true
source .venv/bin/activate

python -m pip install 'dvc[all]'

git init
git checkout -b main
dvc init
# Remote active on this env only, for writing to HTTP redirect below.
dvc remote add -d --local storage s3://dvc-public/remote/get-started
# Actual remote for generated project (read-only). Redirect of S3 bucket above.
dvc remote add -d storage https://remote.dvc.org/get-started
tag_tick
cp -ar "${HERE}"/code-checkpoints/00-common/. "${REPO_PATH}"/
git add "${REPO_PATH}/"*
git commit -m "Initialized DVC and Configured Remote"
git tag -a "init" -m "Initialized DVC and Remote"

test -d data/fashion-mnist || mkdir -p data/fashion-mnist

dvc import https://github.com/iterative/dataset-registry \
           fashion-mnist/raw -o data/fashion-mnist/raw

tag_tick
git add data/fashion-mnist/raw.dvc data/fashion-mnist/.gitignore
git commit -m "Add Fashion-MNIST data"
git tag -a "data" -m "Fashion-MNIST data file added."
dvc push

cp -arf "${HERE}"/code-checkpoints/00-common/. "${REPO_PATH}"/
cp -arf "${HERE}"/code-checkpoints/basic/. "${REPO_PATH}"/
pip install -r "${REPO_PATH}"/requirements.txt

tag_tick
git add .
git commit -m "Added 'checkpoint: true' to the params file"
git tag -a "basic" -m "checkpoint: true example"


cp -arf "${HERE}"/code-checkpoints/00-common/. "${REPO_PATH}"/
cp -arf "${HERE}"/code-checkpoints/signal-file/. "${REPO_PATH}"/
pip install -r "${REPO_PATH}"/requirements.txt

tag_tick
git add .
git commit -m "Modified train.py for signal files"
git tag -a "signal-file" -m "Checkpoints: Signal file example"


cp -arf "${HERE}"/code-checkpoints/00-common/. "${REPO_PATH}"/
cp -arf "${HERE}"/code-checkpoints/python-api/. "${REPO_PATH}"/
pip install -r "${REPO_PATH}"/requirements.txt

tag_tick
git add .
git commit -m "Using make_checkpoint() in the callback"
git tag -a "python-api" -m "Checkpoints: Python API example"


cp -arf "${HERE}"/code-checkpoints/00-common/. "${REPO_PATH}"/
cp -arf "${HERE}"/code-checkpoints/dvclive/. "${REPO_PATH}"/
pip install -r "${REPO_PATH}"/requirements.txt

tag_tick
git add .
git commit -m "DVClive modifications"
git tag -a "dvclive" -m "Checkpoints: DVClive stage added"

popd

unset REPO_PATH
