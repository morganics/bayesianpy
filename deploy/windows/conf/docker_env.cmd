@echo off
SET DOCKER_TLS_VERIFY=1
SET DOCKER_HOST=tcp://192.168.99.100:2376
SET DOCKER_MACHINE_NAME=default
SET DOCKER_CERT_PATH=C:\Users\imorgan.admin\.docker\machine\machines\%DOCKER_MACHINE_NAME%
@echo on