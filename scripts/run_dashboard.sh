#!/usr/bin/env bash
set -euo pipefail

echo "=== Pollen & Pain: Launching Dashboard ==="
streamlit run src/dashboard/app.py --server.port 8501
