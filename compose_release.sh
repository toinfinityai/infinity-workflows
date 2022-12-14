#!/bin/bash

# Composes code into .zip file

CODE_REV=$(git describe --dirty --always)
VERSION=$(poetry version -s)-$CODE_REV
RELEASE_NAME="infinity-tutorials-${VERSION}"

set -e

BASE_DIR=$(pwd)

# Release directory
RELEASE_DIR="${BASE_DIR}/release"
rm -rf "${RELEASE_DIR}"
mkdir -p "${RELEASE_DIR}"

# Temporary directory
TMP_DIR=${BASE_DIR}/tmp/compose
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"

# Compose code with all submodules into one .zip file
git archive -o "${TMP_DIR}/infinity-tutorials.zip" HEAD
cd "${TMP_DIR}"
unzip infinity-tutorials.zip -d infinity-tutorials
rm -rf infinity-tutorials/infinity-tools

INFINITY_TOOLS_DIR=${BASE_DIR}/infinity-tools
INFINITY_CORE_DIR="${INFINITY_TOOLS_DIR}/infinity-core"

WHEELS_DIR=${TMP_DIR}/infinity-tutorials/wheels

cd "${INFINITY_TOOLS_DIR}"
pip wheel --no-deps --wheel-dir "${WHEELS_DIR}" .

cd "${INFINITY_CORE_DIR}"
pip wheel --no-deps --wheel-dir "${WHEELS_DIR}" .

cd "${TMP_DIR}"
rm -rf infinity-tutorials.zip
zip -r infinity-tutorials.zip infinity-tutorials

cd "${BASE_DIR}"
cp "${TMP_DIR}/infinity-tutorials.zip" "${RELEASE_DIR}/${RELEASE_NAME}.zip"

# Remove temporary files
rm -rf "${TMP_DIR}"
