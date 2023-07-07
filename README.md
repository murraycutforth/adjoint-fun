# adjoint-fun

## Getting Started

The environment for running this project is based on the dolfin-adjoint docker images. 
This is described on `http://www.dolfin-adjoint.org/en/latest/download/index.html`- the dolfin adjoint image is called `quay.io/dolfinadjoint/pyadjoint:2019.1.0`. 

Then, inside the `scripts` directory there are various `run_docker.sh` shell scripts, which contain the `docker run` command with appropriate arguments.

## TODO

- Update run scripts so they are machine-independent (similar to run_docker_interactive)
- Fix unit tests, which need updating since codebase has changed
- Experiment with Burgers'-like equation
