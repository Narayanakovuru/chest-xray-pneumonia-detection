#!/usr/bin/env bash
set -e

echo "Running Unit and Integration Test Suite..."

# Execute pytest on target tests folder
python -m pytest tests/ -v
