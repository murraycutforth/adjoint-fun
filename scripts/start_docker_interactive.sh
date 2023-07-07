cd ..
WD=$(pwd)
CONTAINER_DIR=/adjoint-fun

docker run -it --rm -v $WD:$CONTAINER_DIR -w $CONTAINER_DIR -e PYTHONPATH=$CONTAINER_DIR --entrypoint "/bin/bash" quay.io/dolfinadjoint/pyadjoint:2019.1.0
