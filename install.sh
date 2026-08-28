#!/usr/bin/env bash
# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP
# and may only be used strictly in accordance with the applicable license terms.
# By expressly accepting such terms or by downloading, installing, activating
# and/or otherwise using the software, you are agreeing that you have read, and
# that you agree to comply with and are bound by, such license terms.
# If you do not agree to be bound by the applicable license terms, then you may
# not retain, install, activate or otherwise use the software.
#

set -euo pipefail

APP_NAME="robotics-visionai-example"
APP_DIR="${RVE_APP_DIR:-/usr/share/${APP_NAME}}"
VENV="${APP_DIR}/venv"
WHEEL_DIR="${RVE_WHEEL_DIR:-/usr/share/python-wheels}"
WHEEL_GLOB="robotics_visionai_example-*-py3-none-any.whl"


info()    { printf '[%s] %s\n'  "${APP_NAME}" "$*"; }
warning() { printf '[%s] WARNING: %s\n' "${APP_NAME}" "$*" >&2; }
fail()    { printf '[%s] ERROR: %s\n'   "${APP_NAME}" "$*" >&2; exit 1; }

if [ "$(id -u)" -ne 0 ] && [ -z "${RVE_APP_DIR:-}" ]; then
    fail "must run as root (the venv is created under ${APP_DIR}). Try: sudo $0"
fi

mkdir -p "${APP_DIR}" 2>/dev/null || \
    fail "cannot create ${APP_DIR} (permission denied?)."
[ -w "${APP_DIR}" ] || fail "${APP_DIR} is not writable."


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

command -v uv >/dev/null 2>&1 || \
    fail "'uv' not found in PATH.."

# shellcheck disable=SC2086  # deliberate glob expansion
WHEEL=$(ls -1 ${WHEEL_DIR}/${WHEEL_GLOB} 2>/dev/null | sort | tail -n 1 || true)
[ -n "${WHEEL}" ] || \
    fail "no ${WHEEL_GLOB} found in ${WHEEL_DIR}. Reinstall ${APP_NAME}."

info "using wheel: ${WHEEL}"

# ---------------------------------------------------------------------------
# Create the virtual environment
# ---------------------------------------------------------------------------

# Remove any previous venv so a re-run is a clean rebuild
if [ -d "${VENV}" ]; then
    info "removing existing venv at ${VENV}"
    rm -rf "${VENV}"
fi

info "creating venv (${VENV})"
uv venv --system-site-packages "${VENV}"

# ---------------------------------------------------------------------------
# Install the application and its dependencies
# ---------------------------------------------------------------------------

info "installing ${APP_NAME} and its dependencies (requires network access)"
if ! VIRTUAL_ENV="${VENV}" uv pip install --python "${VENV}/bin/python" \
        --no-progress "${WHEEL}"; then
    warning "dependency installation failed."
    warning "The dependency wheels are downloaded from PyPI and are not"
    warning "shipped in the image, so this step needs working network access."
    warning "Check connectivity (and any proxy settings) and re-run:"
    warning "  sudo ${APP_DIR}/install.sh"
    exit 1
fi

[ -x "${VENV}/bin/${APP_NAME}" ] || \
    fail "install completed but ${VENV}/bin/${APP_NAME} is missing."

info "installation complete."
info ""
info "Installation directory: ${APP_DIR}"
info "  venv/       Python virtual environment"
info "  install.sh  this script (re-run to rebuild the venv)"
info "  README.md   full documentation"
info ""
info "Run it with:"
info "  ${APP_NAME} --backend ARA --task detection    # NXP ARA-240, YOLOv8"

info "  ${APP_NAME} --backend NEUTRON --task pose     # i.MX Neutron NPU, MoveNet"
info "  ${APP_NAME} --doctor                          # check the environment"
info ""
info "Then browse to http://<board-ip>:5010"
info ""
info "Models are not included. See ${APP_DIR}/README.md for instructions on obtaining the models."
