@pushd %~dp0
CALL docker save -o ".\docker_image.tar" docker_image:latest
@popd