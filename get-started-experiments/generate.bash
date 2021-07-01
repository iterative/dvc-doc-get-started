#!/usr/bin/env bash

set -veux

HERE="$( cd "$(dirname "$0")" ; pwd -P )"
export HERE
PROJECT_NAME="get-started-experiments"
REPO_NAME="$(date +%F-%H-%M-%S)"
export REPO_NAME

export REPO_ROOT="${HERE}/build/${REPO_NAME}"

# Count the number of git tag calls in this repository
NUM_TAGS=$(grep 'git tag' ${HERE}/generate-* | wc -l)
# Start a bit more in the past
TOTAL_TAGS=$(( NUM_TAGS + 10 ))

export STEP_TIME=$(( RANDOM + 50000 ))
export TAG_TIME=$(( $(date +%s) - ( TOTAL_TAGS * STEP_TIME ) ))

export GIT_AUTHOR_NAME="Olivaw Owlet"
export GIT_AUTHOR_EMAIL="64868532+iterative-olivaw@users.noreply.github.com"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"

tag_tick() {
  export TAG_TIME=$(( TAG_TIME + STEP_TIME ))
  export GIT_AUTHOR_DATE=${TAG_TIME}
  export GIT_COMMITTER_DATE=${TAG_TIME}
}

export -f tag_tick

if [[ -d "$REPO_ROOT" ]]; then
    echo "Repo $REPO_ROOT already exists, please remove it first."
    exit 1
fi

mkdir -p "${REPO_ROOT}"
pushd "${REPO_ROOT}"


add_main_pipeline() {
    mkdir -p models

    dvc stage add -n train \
                -d data/images/ \
                -d src/train.py \
                -p model.conv_units \
                -p train.epochs \
                -o models/model.h5 \
                --plots-no-cache logs.csv \
                --metrics-no-cache metrics.json \
                python3 src/train.py

}

export REPO_PATH="${REPO_ROOT}/${PROJECT_NAME}"

mkdir -p "$REPO_PATH"
pushd "${REPO_PATH}"

virtualenv -p python3 .venv
export VIRTUAL_ENV_DISABLE_PROMPT=true
source .venv/bin/activate
echo '.venv/' > .gitignore
pip install 'dvc[all]'

git init
git checkout -b main
cp $HERE/code-experiments/README.md "${REPO_PATH}" 
cp $HERE/code-experiments/.gitignore "${REPO_PATH}"
tag_tick
git add .gitignore README.md
git commit -m "Initialized Git"
git tag -a "git-init" -m "Initialized Git"

cp -r "${HERE}"/code-experiments/src .
cp "${HERE}"/code-experiments/requirements.txt .
cp "${HERE}"/code-experiments/params.yaml .
pip install -r "${REPO_PATH}"/requirements.txt
tag_tick
git add .
git commit -m "Added source and params"
git tag -a "source-code" -m "Added source code and parameters"

test -d data/ || mkdir -p data/
dvc get https://github.com/iterative/dataset-registry \
        fashion-mnist/images.tar.gz -o data/images.tar.gz

pushd data
tar -xvzf images.tar.gz 
rm -f images.tar.gz 
popd 

# WARNING: We don't add images.tar.gz to neither Git nor DVC here
# git add . operation will add all 70000 images to the repository

# Tutorial should start here
dvc init

tag_tick
git add .dvc
git commit -m "Initialized DVC"
git tag -a "dvc-init" -m "Initialized DVC"


dvc add data/images
tag_tick
git add data/images.dvc data/.gitignore
git commit -m "Added Fashion-MNIST images"
git tag -a "added-data" -m "Fashion-MNIST data file added."

tag_tick
add_main_pipeline
git add dvc.yaml 
git commit -m "Added experiments pipeline"
git tag -a "created-pipeline" -m "Experiments pipeline added."

dvc exp run
tag_tick
echo "model.h5" >> models/.gitignore
git add models/.gitignore data/.gitignore dvc.lock logs.csv metrics.json 
git commit -m "Baseline experiment run"
git tag -a "baseline-experiment" -m "Baseline experiment"

dvc exp run -n cnn-32 --queue -S conv_units=32
dvc exp run -n cnn-64 --queue -S conv_units=64
dvc exp run -n cnn-96 --queue -S conv_units=96
dvc exp run -n cnn-128 --queue -S conv_units=128

dvc exp run --run-all --jobs 2

dvc exp show --no-pager

git status 

PUSH_SCRIPT="${REPO_ROOT}/push-${PROJECT_NAME}.bash"

cat > "${PUSH_SCRIPT}" <<EOF
#!/usr/bin/env bash

set -veux

# The Git repo generated by this script is intended to be published on
# https://github.com/iterative/${PROJECT_NAME}.git Make sure the Github repo
# exists first and that you have appropriate write permissions.

pushd ${REPO_PATH}

dvc remote add --default storage s3://dvc-public/code/${PROJECT_NAME}/
dvc push 

git remote add origin "git@github.com:iterative/${PROJECT_NAME}.git"
  # Delete all tags in the remote
for tag in \$(git ls-remote --tags origin | grep -v '{}$' | cut -c 52-) ; do 
    git push -v origin --delete \${tag}
done
git push --force origin --all --follow-tags
dvc exp list --all --names-only | xargs -n 1 dvc exp push origin
popd
EOF

chmod u+x "${PUSH_SCRIPT}"

popd

cat << EOF
##################################
### REPOSITORY GENERATION DONE ###
##################################

Repositories are in: 

${REPO_ROOT}

Push scripts are written to: 
$(ls -1 ${REPO_ROOT}/*.bash)

You may remove the generated repo with:

$ rm -fR ${REPO_ROOT}
EOF

unset HERE
unset PROJECT_NAME
unset REPO_NAME
unset REPO_ROOT
unset STEP_TIME
unset TAG_TIME
unset GIT_AUTHOR_NAME
unset GIT_AUTHOR_EMAIL
unset GIT_AUTHOR_DATE
unset GIT_COMMITTER_NAME
unset GIT_COMMITTER_EMAIL
unset GIT_COMMITTER_DATE
unset tag_tick
