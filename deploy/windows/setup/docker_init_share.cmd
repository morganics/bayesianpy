@pushd %~dp0
@REM this sets up the shared folder on the local machine. 
@REM NOTE: change any dirs required in conf/conf_env
CALL docker-machine stop %DOCKER_MACHINE_NAME%
CALL vbox_share.cmd

@REM start up the docker-machine
CALL docker-machine start %DOCKER_MACHINE_NAME%
@popd