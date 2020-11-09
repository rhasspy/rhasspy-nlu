#!/usr/bin/env bash
set -e

if [[ -z "${PIP_INSTALL}" ]]; then
    PIP_INSTALL='install'
fi

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

# -----------------------------------------------------------------------------

venv="${src_dir}/.venv"
download="${src_dir}/download"

# -----------------------------------------------------------------------------

function maybe_download {
    if [[ ! -s "$2" ]]; then
        mkdir -p "$(dirname "$2")"
        curl -sSfL -o "$2" "$1" || { echo "Can't download $1"; exit 1; }
        echo "$1 => $2"
    fi
}

num2words_file="${download}/num2words_0.5.10-sv.tar.gz"
num2words_url='https://github.com/rhasspy/num2words/archive/v0.5.10-sv.tar.gz'

maybe_download "${num2words_url}" "${num2words_file}"

# -----------------------------------------------------------------------------

: "${PYTHON=python3}"

# Create virtual environment
echo "Creating virtual environment at ${venv}"
rm -rf "${venv}"
"${PYTHON}" -m venv "${venv}"
source "${venv}/bin/activate"

# Install Python dependencies
echo "Installing Python dependencies"
pip3 ${PIP_INSTALL} --upgrade pip
pip3 ${PIP_INSTALL} --upgrade wheel setuptools

if [[ -f "${num2words_file}" ]]; then
    pip3 ${PIP_INSTALL} "${num2words_file}"
fi

pip3 ${PIP_INSTALL} -r requirements.txt

# Optional development requirements
pip3 ${PIP_INSTALL} -r requirements_dev.txt || \
    echo "Failed to install development requirements"

# -----------------------------------------------------------------------------

echo "OK"
