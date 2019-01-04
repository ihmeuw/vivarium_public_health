#!/bin/bash

source builds/venv/bin/activate

# travis venv tests override python
PYTHON=${PYTHON:-python}
if [ -n "$PYTHON_OPTS" ]; then
  PYTHON="${PYTHON} $PYTHON_OPTS"
fi
PIP=${PIP:-pip}

$PIP install -v .[test,doc]

pytest
