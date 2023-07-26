#!/bin/bash

sudo apt update
sudo apt install -y --no-install-recommends cuda-toolkit-11-8

# Install poetry
pipx install poetry==1.3.2 --pip-args '--no-cache-dir --force-reinstall'

# Install project dependencies
poetry install
