#!/bin/bash

# Define variables
branch_name=$1

vivarium_branch_name='main'
branch_name_to_check=${branch_name}
iterations=0

while [ "$branch_name_to_check" != "$vivarium_branch_name" ] && [ $iterations -lt 20 ]
do
  echo "Checking for vivarium branch: '${branch_name_to_check}'"
  if
    git ls-remote --exit-code \
    --heads https://github.com/ihmeuw/vivarium.git ${branch_name_to_check} == "0"
  then
    vivarium_branch_name=${branch_name_to_check}
    echo "Found matching branch: ${vivarium_branch_name}"
  else
    echo "Could not find upstream branch '${branch_name_to_check}'. Finding parent branch."
    branch_name_to_check="$( \
      git show-branch -a \
      | grep '\*' \
      | grep -v `git rev-parse --abbrev-ref HEAD` \
      | head -n1 \
      | sed 's/[^\[]*//' \
      | awk 'match($0, /\[[a-zA-Z0-9\/.-]+\]/) { print substr( $0, RSTART+1, RLENGTH-2 )}' \
      | sed 's/^origin\///' \
    )"
    if [ -z "$branch_name_to_check" ]; then
      echo "Could not find upstream branch. Will use released version."
      branch_name_to_check="main"
    fi
    echo "Checking out branch: ${branch_name_to_check}"
    git checkout ${branch_name_to_check}
    iterations=$((iterations+1))
  fi
  done
  echo "vivarium_branch_name=${vivarium_branch_name}" >> $GITHUB_ENV
  git checkout ${branch_name}