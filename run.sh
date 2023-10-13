#! /bin/bash

CONTEXT_FILE=${1}
TEST_FILE=${2}
OUTPUT_FILE=${3}

python inference.py ${CONTEXT_FILE} ${TEST_FILE} ${OUTPUT_FILE}

