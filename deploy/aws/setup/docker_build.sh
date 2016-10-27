#!/bin/bash

dir=$(dirname $(readlink /proc/$$/fd/255))

pushd dir

# this is for sharing data over instances of the webserver.. might not be required.. but..
if [ ! -d "/data" ]; then
  mkdir "/data"
fi

if [ ! -d "./scripts" ]; then
  mkdir "./scripts"
fi

if [ -d "./scripts/bayespy" ]; then
    pushd "./scripts/bayespy"
    git fetch https://github.com/morganics/BayesPy.git
    popd
else
    git clone https://github.com/morganics/BayesPy.git "./scripts/bayespy"
fi

docker build -t docker_image .
popd