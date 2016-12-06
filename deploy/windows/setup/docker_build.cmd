@pushd %~dp0
@echo off
REM checkout a particular tagged version.
IF NOT EXIST ./scripts mkdir ./scripts

IF EXIST "./scripts/bayesianpy" (
pushd "./scripts/bayesianpy"
CALL git fetch https://github.com/morganics/BayesianPy.git
popd
) else (
CALL git clone https://github.com/morganics/BayesianPy.git "./scripts/bayesianpy"
)

@echo on
CALL docker build -t docker_image .
@popd