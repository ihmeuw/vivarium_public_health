#!/bin/bash

uname -a
free -m
df -h
ulimit -a
mkdir builds
pushd builds

# Build into our own virtualenv
pip install -U virtualenv

virtualenv --python=python venv

source venv/bin/activate
python -V

pip install --upgrade pip setuptools

# For push builds, TRAVIS_BRANCH is the branch name
# For builds triggered by PR, TRAVIS_BRANCH is the branch targeted by the PR
branch=$TRAVIS_BRANCH
echo branch ${branch}

# Look for branch of same name in upstream repositories
# when this is develop it should be present.
# When it isn't develop, this may or may not exist
upstream_branch_exists="$(git ls-remote --heads https://github.com/ihmeuw/vivarium.git ${branch})"

# if there is a match for upstream, use that, else fall back to develop
# this is redundant only for a PR into develop.
if [ -z "${upstream_branch}" ]  # checks if empty
then
    branch=develop
else
    echo upstream branch found ${upstream_branch}
fi

# clone and install upstream stuff
git clone --branch=$branch https://github.com/ihmeuw/vivarium.git
pushd vivarium
pip install .
popd

popd  # is this right ?

pip install .[test,docs]
