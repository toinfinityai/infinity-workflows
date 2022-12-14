#!/bin/bash

# Runs jupyter notebooks server for tutorials

export PYTHONPATH="$(pwd)"
jupyter notebook
