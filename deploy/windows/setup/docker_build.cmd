@pushd %~dp0
@echo off
REM checkout a particular tagged version.
IF NOT EXIST ./scripts mkdir ./scripts

IF EXIST "./scripts/bayespy" (
pushd "./scripts/bayespy"
CALL git fetch https://github.com/morganics/BayesPy.git
popd
) else (
CALL git clone https://github.com/morganics/BayesPy.git "./scripts/bayespy"
)

@echo on
CALL docker build -t docker_image .
@popd