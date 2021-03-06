#!/bin/bash
# See https://dvc.org/doc/tutorials/get-started

# Setup script env:
#   e   Exit immediately if a command exits with a non-zero exit status.
#   u   Treat unset variables as an error when substituting.
#   x   Print commands and their arguments as they are executed.
set -eux

HERE="$( cd "$(dirname "$0")" ; pwd -P )"
REPO_NAME="example-get-started"
REPO_PATH="$HERE/build/$REPO_NAME"

if [ -d "$REPO_PATH" ]; then
  echo "Repo $REPO_PATH already exists, please remove it first."
  exit 1
fi

TOTAL_TAGS=15
STEP_TIME=100000
BEGIN_TIME=$(( $(date +%s) - ( ${TOTAL_TAGS} * ${STEP_TIME}) ))
export TAG_TIME=${BEGIN_TIME}
export GIT_AUTHOR_DATE=${TAG_TIME}
export GIT_COMMITTER_DATE=${TAG_TIME}
tick(){
  export TAG_TIME=$(( ${TAG_TIME} + ${STEP_TIME} ))
  export GIT_AUTHOR_DATE=${TAG_TIME}
  export GIT_COMMITTER_DATE=${TAG_TIME}
}

export GIT_AUTHOR_NAME="Olivaw Owlet"
export GIT_AUTHOR_EMAIL="64868532+iterative-olivaw@users.noreply.github.com"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"

mkdir -p $REPO_PATH
pushd $REPO_PATH

virtualenv -p python3 .venv
export VIRTUAL_ENV_DISABLE_PROMPT=true
source .venv/bin/activate
echo '.venv/' > .gitignore

pip install gitpython
pip install "git+https://github.com/iterative/dvc#egg=dvc[all]"

git init
cp $HERE/code/README.md .
git add .
tick
git commit -m "Initialize Git repository"
git tag -a "0-git-init" -m "Git initialized."

dvc init
tick
git commit -m "Initialize DVC project"
git tag -a "1-dvc-init" -m "DVC initialized."

mkdir data
dvc get https://github.com/iterative/dataset-registry \
  get-started/data.xml -o data/data.xml
dvc add data/data.xml --desc "Initial XML StackOverflow dataset (raw data)"
git add data/.gitignore data/data.xml.dvc
tick
git commit -m "Add raw data"
git tag -a "2-track-data" -m "Data file added."

# Remote active on this env only, for writing to HTTP redirect below.
dvc remote add -d --local storage s3://dvc-public/remote/get-started
# Actual remote for generated project (read-only). Redirect of S3 bucket above.
dvc remote add -d storage https://remote.dvc.org/get-started
git add .
tick
git commit -m "Configure default remote"
git tag -a "3-config-remote" -m "Read-only remote storage configured."
dvc push

rm data/data.xml data/data.xml.dvc
dvc import https://github.com/iterative/dataset-registry \
  get-started/data.xml -o data/data.xml \
  --desc "Imported raw data (tracks source updates)"
git add data/data.xml.dvc
tick
git commit -m "Import raw data (overwrite)"
git tag -a "4-import-data" -m "Data file overwritten with an import."
dvc push

wget https://code.dvc.org/get-started/code.zip
unzip code.zip
rm -f code.zip
pip install -r src/requirements.txt
git add .
tick
git commit -m "Add source code files to repo"
git tag -a "5-source-code" -m "Source code added."


dvc run -n prepare \
  -p prepare.seed,prepare.split \
  -d src/prepare.py -d data/data.xml \
  -o data/prepared \
  python src/prepare.py data/data.xml
git add data/.gitignore dvc.yaml dvc.lock
tick
git commit -m "Create data preparation stage"
git tag -a "6-prepare-stage" -m "First pipeline stage (data preparation) created."
dvc push


dvc run -n featurize \
  -p featurize.max_features,featurize.ngrams \
  -d src/featurization.py -d data/prepared \
  -o data/features \
  python src/featurization.py \
  data/prepared data/features
git add data/.gitignore dvc.yaml dvc.lock
dvc run -n train \
  -p train.seed,train.n_est,train.min_split \
  -d src/train.py -d data/features \
  -o model.pkl \
  python src/train.py data/features model.pkl
git add .gitignore dvc.yaml dvc.lock
tick
git commit -m "Create ML pipeline stages"
git tag -a "7-ml-pipeline" -m "ML pipeline created."
dvc push

dvc run -n evaluate \
  -d src/evaluate.py -d model.pkl -d data/features \
  -M scores.json \
  --plots-no-cache prc.json \
  --plots-no-cache roc.json \
  python src/evaluate.py model.pkl data/features scores.json prc.json roc.json
dvc plots modify prc.json -x recall -y precision
dvc plots modify roc.json -x fpr -y tpr
git add .gitignore dvc.yaml dvc.lock prc.json roc.json scores.json
tick
git commit -m "Create evaluation stage"
git tag -a "baseline-experiment" -m "Baseline experiment evaluation"
git tag -a "8-evaluation" -m "Baseline evaluation stage created."
dvc push

sed -e "s/max_features: 500/max_features: 1500/" -i".bkp" params.yaml
sed -e "s/ngrams: 1/ngrams: 2/" -i".bkp" params.yaml
dvc repro train
tick
git commit -am "Reproduce model using bigrams"
git tag -a "9-bigrams-model" -m "Model retrained using bigrams."

dvc repro evaluate
tick
git commit -am "Evaluate bigrams model"
git tag -a "bigrams-experiment" -m "Bigrams experiment evaluation"
git tag -a "10-bigrams-experiment" -m "Evaluated bigrams model."
dvc push

dvc exp run --set-param featurize.max_features=3000
dvc exp run --queue --set-param train.min_split=8
dvc exp run --queue --set-param train.min_split=64
dvc exp run --queue --set-param train.min_split=2 --set-param train.n_est=100
dvc exp run --queue --set-param train.min_split=8 --set-param train.n_est=100
dvc exp run --queue --set-param train.min_split=64 --set-param train.n_est=100
dvc exp run --run-all -j 2
# Apply best experiment
dvc exp apply $(dvc exp show --no-pager --sort-by avg_prec | tail -n 2 | head -n 1 | grep -o 'exp-\w*')
git add .github/workflows/cml.yaml
tick
git commit -am "Run experiments tuning random forest params"
git tag -a "random-forest-experiments" -m "Run experiments to tune random forest params"
git tag -a "11-random-forest-experiments" -m "Tuned random forest classifier."
dvc push

popd

unset TAG_TIME
unset GIT_AUTHOR_DATE
unset GIT_COMMITTER_DATE
unset GIT_AUTHOR_NAME
unset GIT_AUTHOR_EMAIL
unset GIT_COMMITTER_NAME
unset GIT_COMMITTER_EMAIL

echo "`cat <<EOF-

The Git repo generated by this script is intended to be published on
https://github.com/iterative/example-get-started. Make sure the Github repo
exists first and that you have appropriate write permissions.

To create it with https://hub.github.com/ for example, run:

hub create iterative/example-get-started -d "Get Started DVC project" \
-h "https://dvc.org/doc/get-started"

If the Github repo already exists, run these commands to rewrite it:

cd build/example-get-started
git remote add origin git@github.com:iterative/example-get-started.git
git push --force origin master
git push --force origin --tags

Run these to drop and then rewrite the experiment references on the repo:

git ls-remote origin "refs/exps/*" | awk '{print $2}' | xargs -n 1 git push -d origin
dvc exp list --all --names-only | xargs -n 1 dvc exp push origin

Finally, return to the directory where you started:

cd ../..

You may remove the generated repo with:

rm -fR build

`"
