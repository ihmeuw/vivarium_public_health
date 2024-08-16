#!/bin/bash

Help()
{
  # Reset OPTIND so help can be invoked multiple times per shell session.
  OPTIND=1
   # Display Help
   echo "Script to automatically create and validate conda environments."
   echo
   echo "Syntax: source get_upstream_branch.sh [-h|b|d]"
   echo "options:"
   echo "h     Print this Help."
   echo "b     Name of the local branch."
   echo "d     'yes' if in debug mode; else 'no'."
}

# Define variables
branch_name="main"
debug="no"

# Process input options
while getopts ":hbd:" option; do
   case $option in
      h) # display help
         Help
         return;;
      b) # Name of the local branch
         branch_name=$OPTARG;;
      d) # Debug mode
          debug=$OPTARG;;
     \?) # Invalid option
         echo "Error: Invalid option"
         return;;
   esac
done

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
  if [ $debug == 'no' ]; then
    echo "vivarium_branch_name=${vivarium_branch_name}" >> $GITHUB_ENV
  fi
  git checkout ${branch_name}