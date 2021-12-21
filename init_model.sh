#!/bin/bash

mkdir initial_model

FILE_NAME=initial_model.npz

cd client

python3 init_model.py

echo done