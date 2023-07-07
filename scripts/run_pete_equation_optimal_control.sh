#!/bin/bash

# Note: this script hard codes the number of OpenBLAS threads

WD=$(cd .. && pwd)
CONTAINER_DIR=/adjoint-fun

docker run -it --rm -v $WD:/$CONTAINER_DIR -w $CONTAINER_DIR -e PYTHONPATH=$CONTAINER_DIR --entrypoint "/bin/bash" -e OPENBLAS_NUM_THREADS=4 quay.io/dolfinadjoint/pyadjoint:2019.1.0 -c "python3 src/adjoint/advection_diffusion_control_static_velocity_and_f.py"
