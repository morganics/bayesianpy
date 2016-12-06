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

if [ -d "./scripts/bayesianpy" ]; then
    pushd "./scripts/bayesianpy"
    git fetch https://github.com/morganics/BayesianPy.git
    popd
else
    git clone https://github.com/morganics/BayesianPy.git "./scripts/bayesianpy"
fi

docker build -t docker_image .
popd