@echo off
REM This is the folder that you want to share on your local machine.
SET HOST_SHARE_FOLDER=C:/Users/imorgan.admin/share/sbgalerts

SET HOST_VOLUME_PATH=/Users
SET VOLUME_PATH=/Users
SET DOCKER_CMD_ARGS=--volume %HOST_VOLUME_PATH%:%VOLUME_PATH% --rm -e SBG_APP_ID -e SBG_APP_SECRET docker_image:latest
@echo on