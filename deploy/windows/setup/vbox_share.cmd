CALL "c:\program files\oracle\virtualbox\vboxmanage" sharedfolder remove %DOCKER_MACHINE_NAME% -name Users
CALL "c:\program files\oracle\virtualbox\vboxmanage" sharedfolder add %DOCKER_MACHINE_NAME% -name Users -hostpath %HOST_SHARE_FOLDER% --automount
