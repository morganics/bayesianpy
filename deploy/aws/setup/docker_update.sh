#!/bin/bash
git fetch --all
git reset --hard origin/master
sudo /bin/bash ./docker_build.sh