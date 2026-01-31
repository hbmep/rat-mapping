SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

PY ?= python3.11
VENV := .venv
PIP := $(VENV)/bin/python -m pip
PIP_NO_CACHE := --no-cache-dir

.PHONY: base env

base:
	rm -rf $(VENV) build
	@echo "Creating virtual environment with $(PY)..."
	$(PY) -m venv $(VENV)
	@echo "Upgrading pip..."
	$(PIP) install --upgrade pip
	@echo "Purging pip cache..."
	$(PIP) cache purge

env: base
	@echo "Installing package..."
	$(PIP) install $(PIP_NO_CACHE) -e .
	$(PIP) install $(PIP_NO_CACHE) -e ../hbmep
