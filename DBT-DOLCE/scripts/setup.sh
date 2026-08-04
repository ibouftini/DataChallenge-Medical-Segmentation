#!/usr/bin/env bash
# Full environment setup:
#   1. Clones the DOLCE repo (provides guided_diffusion + LEAP source)
#   2. Installs the conda environment
#   3. Builds and installs LEAP from the DOLCE-bundled source
#   4. Installs guided_diffusion in editable mode

set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# 1. Clone DOLCE
echo "[1/4] Cloning DOLCE ..."
if [ ! -d "$ROOT_DIR/external/DOLCE" ]; then
    git clone https://github.com/wustl-cig/DOLCE.git "$ROOT_DIR/external/DOLCE"
else
    echo "  Already present, skipping."
fi

# 2. Conda environment
echo "[2/4] Creating conda environment 'dbt-dolce' ..."
conda env create -f "$ROOT_DIR/environment.yml" --force

# 3. LEAP
# DOLCE's install_envs.sh creates its own "dolce" conda env; we don't call it.
# Instead we build the LEAP_torch extension directly into our "dbt-dolce" env.
# The LEAP setup.py (which declares py_modules=['LEAP_torch']) lives in
# leap/src, not leap/ itself, so the editable install must run from there.
echo "[3/4] Building LEAP_torch into dbt-dolce ..."
LEAP_SRC="$ROOT_DIR/external/DOLCE/leap/src"
[ -f "$LEAP_SRC/setup.py" ] || { echo "ERROR: LEAP setup.py not found at $LEAP_SRC" >&2; exit 1; }
# leap/src/setup.py imports torch and pybind11 at module top level but ships no
# pyproject.toml declaring them as build requirements. With pip's default PEP 517
# build isolation the build env lacks both, so metadata generation fails. Install
# pybind11 into the env (torch is already there from environment.yml) and build
# with --no-build-isolation so the env's torch + pybind11 are used.
conda run -n dbt-dolce bash -c "
    set -e
    # cuda-toolkit installs nvcc + libs under the env prefix; point CUDA_HOME
    # there so torch's CUDAExtension can locate the toolchain.
    export CUDA_HOME=\"\$CONDA_PREFIX\"
    # CUDA 11.8 needs host GCC <= 11. The env ships conda's GCC 11 toolchain and
    # its activation sets CC/CXX to it; fall back to the conda compiler names if
    # not, and force nvcc to use the same g++ as its host compiler via -ccbin so
    # the build never picks up a newer system GCC.
    export CC=\"\${CC:-\$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc}\"
    export CXX=\"\${CXX:-\$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++}\"
    export NVCC_PREPEND_FLAGS=\"-ccbin \$CXX\"
    cd \"$LEAP_SRC\"
    pip install pybind11
    # Non-editable install: leap/src ships no pyproject.toml, so an editable
    # install falls back to setuptools' 'develop', which re-invokes pip with
    # --use-pep517 and re-enables build isolation -- losing torch and failing
    # with ModuleNotFoundError. A regular install builds in-process with the
    # env's torch. LEAP is an external dependency, so editable buys us nothing.
    pip install . --no-build-isolation
"

# 4. guided_diffusion (editable install from DOLCE)
echo "[4/4] Installing guided_diffusion in editable mode ..."
conda run -n dbt-dolce pip install -e "$ROOT_DIR/external/DOLCE"

# 5. Verify the LEAP build actually imports, then run the projector smoke test.
echo ""
echo "Verifying LEAP import ..."
conda run -n dbt-dolce python -c "from LEAP_torch import Projector; print('  LEAP import OK')"

echo "Running projector smoke test (CPU) ..."
conda run -n dbt-dolce python "$ROOT_DIR/tests/test_projector.py"

echo ""
echo "Setup complete. Activate with:  conda activate dbt-dolce"
echo "Then download weights:          bash scripts/download_weights.sh"
