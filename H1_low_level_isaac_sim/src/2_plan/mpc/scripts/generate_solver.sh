#!/bin/bash
# This shebang guarantees the script is executed with bash, not /bin/sh

# Exit immediately if any command fails
set -e

# --- Read Arguments from CMake ---
VENV_ACTIVATE_SCRIPT=$1
PYTHON_SCRIPT=$2
MODELS_PATH=$3
GEN_CODE_PATH=$4

# --- Print for Debugging ---
echo "--- Wrapper script starting..."
echo "--- Venv Activate Script: ${VENV_ACTIVATE_SCRIPT}"
echo "--- Python Script to Run: ${PYTHON_SCRIPT}"
echo "--- Path to Models:       ${MODELS_PATH}"
echo "--- Code Generation Path: ${GEN_CODE_PATH}"

env -i PATH="/usr/local/bin:/usr/bin:/bin" \
bash -c "
  echo '--- Innerhalb der sauberen Umgebung. Setze Acados-Pfade...'
  export ACADOS_SOURCE_DIR='${ACADOS_SOURCE_DIR}'
  export LD_LIBRARY_PATH='${ACADOS_SOURCE_DIR}/lib'

  echo '--- Aktiviere venv...'
  source '${VENV_ACTIVATE_SCRIPT}'
  echo '--- Führe Python-Skript aus...'
  python3 '${PYTHON_SCRIPT}' --models_path '${MODELS_PATH}' --gen_code_path '${GEN_CODE_PATH}'
"

echo "--- Wrapper script finished successfully."