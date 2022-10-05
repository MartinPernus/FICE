#!/bin/bash

echo "Downloading pretrained models ..."
wget -O checkpoints.tar.xz https://www.dropbox.com/s/qx9mag5hh7tleso/checkpoints.tar.xz?dl=1
echo "Extracting ..."
tar -xf checkpoints.tar.xz -C checkpoints
echo "Done!"
