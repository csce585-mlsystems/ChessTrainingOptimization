#!/bin/bash
set -e
rm -rf onnxruntime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-1.17.3.tgz
tar -xzf onnxruntime-linux-x64-1.17.3.tgz
mv onnxruntime-linux-x64-1.17.3 onnxruntime
