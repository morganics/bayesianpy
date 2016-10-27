@pushd %~dp0
CALL docker load -i ./docker_image.tar
@popd