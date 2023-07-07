"""In this script we solve an advection-diffusion equation constrained optimisation problem.

The control parameter is the velocity field and source term, which are constant in time.

The error functional is the difference with the target image, in addition to some regularisation terms.
"""

from pathlib import Path
import logging
from time import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import fenics as fe
import fenics_adjoint as fa
import moola

from src.plotting_utils.moola import plot_optimisation_convergence, plot_fe_function_comparison
from src.plotting_utils.advection_diffusion import plot_final, plot_timestep_solns, plot_final_param_fields
from src.image_utils.image_utils import image_to_array, numpy_array_to_fenics_fn



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    mesh, output_dir = initial_setup()

    target_fn = construct_target_function(mesh)

    solve_optimal_control(mesh, target_fn, output_dir)

    logger.info("Program complete normally")


def initial_setup() -> Tuple[fa.Mesh, Path]:
    logger.setLevel(logging.INFO)
    fe.set_log_active(False)
    logging.getLogger("FFC").setLevel(logging.WARNING)
    mesh = fa.UnitSquareMesh(100, 100)
    output_dir = Path("output/adjoint/advection_diffusion/control_static_v_and_f_100x100")

    return mesh, output_dir


def construct_target_function(mesh: fa.Mesh) -> fa.Function:
    # Target solution comes from image
    V = fe.FunctionSpace(mesh, "CG", 1)
    image_path = "inputs/rect14.png"
    image = image_to_array(image_path)
    image = image / 10.0  # Rescale image intensities to be appropriate for equation solution
    target_fe = numpy_array_to_fenics_fn(image, V)
    target_fn = fa.Function(V)
    target_fn.interpolate(target_fe)

    return target_fn


def solve_optimal_control(mesh: fa.Mesh,
                          target_fn: fa.Function,
                          output_dir: Path) -> None:
    """
    Function to run optimisation of f for target u
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info('Started solve_optimal_control for advection-diffusion equation')
    logger.info(f'Mesh has {mesh.num_vertices()} vertices')
    logger.info(f'All output will be stored in {output_dir}')

    # Function spaces
    V = fe.FunctionSpace(mesh, "CG", 1)
    W = fe.FunctionSpace(mesh, "DG", 0)
    V_vector = fe.VectorFunctionSpace(mesh, "DG", 0)

    mesh_coarse = fa.UnitSquareMesh(32, 32)  # Use a coarser mesh for the control parameters
    V_coarse = fe.FunctionSpace(mesh_coarse, "DG", 0)
    V_vector_coarse = fe.VectorFunctionSpace(mesh_coarse, "DG", 0)

    # Parameters
    num_steps = 25
    kappa = fa.Constant(0.001)
    T = 0.5
    delta_t = fa.Constant(T / num_steps)

    # Set initial condition to random noise
    np.random.seed(42)
    ic_array = np.random.normal(15, 10, size=(5, 5))
    u_0_fe = numpy_array_to_fenics_fn(ic_array, V)
    u_n = fa.Function(V)
    u_n.interpolate(u_0_fe)

    # Plot IC
    c = fe.plot(u_n)
    fig = plt.gcf()
    fig.colorbar(c, ax=plt.gca())
    fig.savefig(output_dir / "ic.png")

    f_c = fa.interpolate(fa.Expression("0.0", degree=1), V_coarse)  # Zero source term
    a_c = fa.interpolate(fa.Expression(("0.0", "0.0"), degree=3), V_vector_coarse)  # Initialise velocity field to zero

    u = fa.Function(V)
    v = fe.TestFunction(V)
    f = fa.Function(W)
    a = fa.Function(V_vector)

    # Run forward in time to record tape for adjoint method, using fa overloads
    def run_forward(num_steps, timeseries_u: fe.TimeSeries = None):
        logger.info("Starting run_forward")

        u_n.interpolate(u_0_fe)  # Set IC

        time = 0.0

        if timeseries_u is not None:
            timeseries_u.store(u_n.vector(), time)

        for i in range(num_steps):
            time += float(delta_t)

            # Variational problem using current velocity field

            f.interpolate(f_c)
            a.interpolate(a_c)

            # Backwards Euler time discretisation
            # Variational form for advective term, diffusive term, and source term
            F = (v * (u - u_n) / delta_t \
                    + v * fe.dot(a, fe.grad(u)) \
                    + kappa * fe.inner(fe.grad(u), fe.grad(v))\
                    - f * v) * fe.dx

            fa.solve(F == 0, u)

            u_n.assign(u)
            if timeseries_u is not None:
                timeseries_u.store(u_n.vector(), time)

            logger.info(f"Time step complete. Time = {time:.4g}. norm(u) = {fe.norm(u):.3g}")
        logger.info("Time iterations complete")

    run_forward(num_steps=num_steps, timeseries_u=None)

    # Error functional
    J = fa.assemble(0.5 * (fe.inner(u - target_fn, u - target_fn)) * fe.dx)  # Difference with target
    alpha = fa.Constant(1e-6)
    J += fa.assemble(alpha / 2 * fe.inner(a_c, a_c) * fe.dx)
    f_weight = fa.Constant(4e-2)
    J += fa.assemble(f_weight / 2 * fe.inner(f_c, f_c) * fe.dx)
    
    rf = fa.ReducedFunctional(J, [fa.Control(a_c), fa.Control(f_c)])

    # Apply BFGS from moola library to minimise J
    problem = fa.MoolaOptimizationProblem(rf)
    m_moola = moola.DolfinPrimalVectorSet([moola.DolfinPrimalVector(a_c), moola.DolfinPrimalVector(f_c)])
    solver = moola.BFGS(
            problem,
            m_moola, 
            options={'jtol': 1e-9,
                     'gtol': 1e-9,
                     'Hinit': "default",
                     'maxiter': 50,
                     'mem_lim': 10,
                     "line_search": "strong_wolfe",
                     "line_search_options": {"ignore_warnings": True}
                     }
            )

    logger.info('Starting optimisation')
    start = time()
    sol = solver.solve()
    opt_time = time() - start
    logger.info(f'Optimisation complete in {opt_time:.2g}s')

    plot_optimisation_convergence(solver.history['objective'],
                                  output_dir / 'objective_function.png')


    # Run forward problem using optimal value of f
    m_opt = sol['control'].data
    a_c.assign(m_opt[0])
    f_c.assign(m_opt[1])
    
    u_ts_filename = str(output_dir / "timeseries_u")
    timeseries_u = fe.TimeSeries(u_ts_filename)
    run_forward(num_steps=2 * num_steps, timeseries_u=timeseries_u)


    # Finally, create plots of solution

    plot_timestep_solns(V, u_ts_filename, 2 * T, 16, str(output_dir / "timesteps_combined.png"))

    plot_timestep_solns(V, u_ts_filename, 2 * T, 50, str(output_dir / "timesteps_frame.png"), separate_plots=True)

    plot_final(u, f, a, str(output_dir / "optimised_u_final.png"))

    plot_final_param_fields(f, a, str(output_dir / "final_param_fields.png"))

    plot_fe_function_comparison(u, target_fn, 'Simulation', 'Target',
                                output_dir / 'optimised_solution_vs_target.png')


if __name__ == "__main__":
    main()
