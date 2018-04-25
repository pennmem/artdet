#!/usr/bin/env bash
set -e
conda env export --no-builds --file environment.yml
