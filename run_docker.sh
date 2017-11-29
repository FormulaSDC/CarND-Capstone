#!/bin/bash

docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone

pip uninstall pillow
pip install pillow
