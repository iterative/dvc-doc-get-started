#!/bin/bash

set -veux

HERE="$( cd "$(dirname "$0")" ; pwd -P )"
export HERE
REPO_NAME="dvc-get-started-$(date +%F-%H-%M-%S)"
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


if [ -d "$REPO_ROOT" ]; then
    echo "Repo $REPO_ROOT already exists, please remove it first."
    exit 1
fi

mkdir -p "${REPO_ROOT}"
pushd "${REPO_ROOT}"

# Create the main branch 
"${HERE}"/generate-pipelines.bash
# Create experiments branch
"${HERE}"/generate-experiments.bash
# Create checkpoints branch
"${HERE}"/generate-checkpoints.bash

popd


for d in pipelines checkpoints experiments ; do 

  PUSH_SCRIPT="${REPO_ROOT}/push-${d}.bash"
  cat > "${PUSH_SCRIPT}" <<EOF
#!/bin/bash

set -veux

# The Git repo generated by this script is intended to be published on
# https://github.com/iterative/get-started-${d}.git Make sure the Github repo
# exists first and that you have appropriate write permissions.

pushd ${REPO_ROOT}/${d}

git remote add origin "git@github.com:iterative/get-started-${d}.git"
  # Delete all tags in the remote
for tag in \$(git ls-remote --tags origin | grep -v '{}$' | cut -c 52-) ; do 
    git push -v origin --delete \${tag}
done
git push --force origin --all --follow-tags
dvc exp list --all --names-only | xargs -n 1 dvc exp push origin
popd
EOF

  chmod u+x "${PUSH_SCRIPT}"
done

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