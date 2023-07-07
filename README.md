# adjoint-fun

## Demo

In this example, I wanted to "discover" an equation for my Dad's birthday, which was a specific parameterisation of an advection-diffusion equation which would yield a solution at t=0.5 of my Dad's face! The advection equation we are using as a constraint here is given by

$$\frac{\partial u}{\partial t} + \mathbf{a} \cdot \nabla u = \kappa \nabla^2 u + f$$
where $u$ is the quantity of interest, $\mathbf{a}$ represents the constant velocity field, $\kappa$ denotes the diffusion coefficient, $f$ is the source term, and $t$ is the time variable. 

In order to discover this particular parameterisation, I used the adjoint method implemented in `dolfin-adjoint`, to solve a PDE-constrained optimisation problem.
I had to cheat a little bit and add a (heavily regularised) source term in order to get a satisfactory solution. 

The target solution was set equal to this photo of my Dad:
![rect14](https://github.com/murraycutforth/adjoint-fun/assets/11088372/98b3a07d-b995-4717-964b-093977d317e6)

The, we optimised the following error functional:
$$J = \frac{1}{2} \int_\Omega (u(t=T) - d)^2 dx + \frac{\alpha}{2} \int_\Omega \textbf{a}^T\textbf{a} \ dx + \frac{\beta}{2} \int_\Omega f^2 dx$$
where $d$ corresponded to the photo above.

After 50 iterations of L-BFGS, we discovered an equation with the following solution:
![timesteps_frame_99](https://github.com/murraycutforth/adjoint-fun/assets/11088372/f682faa1-7073-4ab4-b74e-d2078d1e7924)

The source term and velocity fields which were discovered look like this:
![final_param_fields](https://github.com/murraycutforth/adjoint-fun/assets/11088372/c30cb432-787a-46a4-880a-ad0a4e6ead20)

And finally, here's a visualisation of the time evolution of the solution:
![output_tiny](https://github.com/murraycutforth/adjoint-fun/assets/11088372/208c2ab0-0460-4b38-a717-68ddd8fbff75)


## Getting Started

The environment for running this project is based on the dolfin-adjoint docker images. 
This is described on `http://www.dolfin-adjoint.org/en/latest/download/index.html`- the dolfin adjoint image is called `quay.io/dolfinadjoint/pyadjoint:2019.1.0`. 

Then, inside the `scripts` directory there are various `run_docker.sh` shell scripts, which contain the `docker run` command with appropriate arguments.

## TODO

- Update run scripts so they are machine-independent (similar to run_docker_interactive)
- Fix unit tests, which need updating since codebase has changed
- Experiment with Burgers'-like equation
