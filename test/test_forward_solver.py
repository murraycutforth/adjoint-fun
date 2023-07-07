import unittest
import logging

import fenics as fe
import numpy as np


logging.getLogger("FFC").setLevel(logging.ERROR)
fe.set_log_active(False)


from src.solver.advection_diffusion import forward_solve_static_f, forward_solve_dynamic_f
from src.solver.utils import compute_vertex_linf
from src.plotting_utils.advection_diffusion import plot_final, plot_timestep_solns


class TestAdvectionDiffusionForward(unittest.TestCase):
    def test_gaussian_source(self):
        # Check images in output/test/advection_diffusion/gaussian to verify approximate correctness
        u_ts_filename = "output/test/advection_diffusion/gaussian/u_ts"

        mesh = fe.UnitSquareMesh(20, 20)
        V = fe.FunctionSpace(mesh, "CG", 1)
        V_vector = fe.VectorFunctionSpace(mesh, "CG", 1)
        u = fe.Function(V, name="Temperature")
        u_prev = fe.interpolate(fe.Expression("0.0", name="u_prev", degree=1), V)

        # Velocity field
        a = fe.interpolate(fe.Expression(("-1.0", "-1.0"), name='Velocity', degree=1), V_vector)

        # Gaussian point source term
        f = fe.interpolate(fe.Expression("exp(-(pow(x[0] - 0.75, 2) + pow(x[1] - 0.75, 2)) / 0.01)", name='Source term', degree=1), V)

        # Thermal diffusivity
        kappa = fe.Constant(0.01)

        # Time step and total time
        delta_t = 0.01
        T = 0.5

        # Call solver
        forward_solve_static_f(u, u_prev, V, a, kappa, f, delta_t, T, u_ts_filename)

        # Final plot output
        plot_final(u, f, a, "output/test/advection_diffusion/gaussian/final.png")
        plot_timestep_solns(V, u_ts_filename, T, 16, "output/test/advection_diffusion/gaussian/u_ts.png")

        # Regression test
        self.assertAlmostEqual(fe.norm(u, "L2"), 0.0286, delta=0.0001)


    def test_mms_one_step(self):
        # Use method of manufactured solutions to check numerical accuracy of a single time step
        mesh = fe.UnitSquareMesh(32, 32)
        V = fe.FunctionSpace(mesh, "CG", 1)
        V_vector = fe.VectorFunctionSpace(mesh, "CG", 1)
        u = fe.Function(V, name="Temperature")

        # Initial condition
        u_prev = fe.interpolate(fe.Expression("0.0", name="u_prev", degree=1), V)

        # Velocity field
        a = fe.interpolate(fe.Expression(("1.0", "1.0"), name='Velocity', degree=1), V_vector)

        # Thermal diffusivity
        kappa = fe.Constant(1.0)

        # Time step and total time
        delta_t = 0.1
        T = delta_t

        # MMS
        cpp_code = "x[0] * x[1] * (1.0 - x[0]) * (1.0 - x[1]) + delta_t * x[1] * (1.0 - x[1]) * (1.0 - 2.0 * x[0]) + delta_t * x[0] * (1.0 - x[0]) * (1.0 - 2.0 * x[1]) - 2.0 * kappa * x[0] * x[1] * delta_t * (2.0 - x[0] - x[1])"
        f = fe.interpolate(fe.Expression(cpp_code, delta_t=delta_t, kappa=kappa, name='Source term', degree=3), V)

        # Call solver
        forward_solve_static_f(u, u_prev, V, a, kappa, f, delta_t, T)

        u_exact = fe.interpolate(fe.Expression("delta_t * x[0] * x[1] * (1.0 - x[0]) * (1.0 - x[1])", delta_t=delta_t, degree=3), V)

        # Compute errors
        error_L2 = fe.errornorm(u, u_exact, "L2")
        error_Linf = compute_vertex_linf(mesh, u, u_exact)

        print(error_L2)
        print(error_Linf)

        plot_final(u, f, a, "output/test/advection_diffusion/mms_onestep/final.png")

    def test_mms_multi_step(self):
        # Use method of manufactured solutions to check numerical accuracy of multiple time steps
        mesh = fe.UnitSquareMesh(32, 32)
        V = fe.FunctionSpace(mesh, "CG", 1)
        V_vector = fe.VectorFunctionSpace(mesh, "CG", 1)
        u = fe.Function(V, name="Temperature")

        # Initial condition
        u_prev = fe.interpolate(fe.Expression("0.0", name="u_prev", degree=1), V)

        # Velocity field
        a = fe.interpolate(fe.Expression(("1.0", "1.0"), name='Velocity', degree=1), V_vector)

        # Thermal diffusivity
        kappa = fe.Constant(1.0)

        # Time step and total time
        delta_t = 0.01
        T = 0.5

        # MMS
        f_exp_cpp_code = "x[0] * x[1] * (1.0 - x[0]) * (1.0 - x[1]) + t * x[1] * (1.0 - x[1]) * (1.0 - 2.0 * x[0]) + t * x[0] * (1.0 - x[0]) * (1.0 - 2.0 * x[1]) - 2.0 * kappa * x[0] * x[1] * t * (2.0 - x[0] - x[1])"
        f = fe.Expression(f_exp_cpp_code, t=delta_t, kappa=kappa, name='Source term', degree=3)

        # Call solver
        logging.basicConfig(level=logging.INFO)  # TODO: temp
        forward_solve_dynamic_f(u, u_prev, V, a, kappa, f, delta_t, T)

        u_exact = fe.interpolate(fe.Expression("x[0] * x[1] * (1.0 - x[0]) * (1.0 - x[1]) * T", T=T, degree=3), V)

        # Compute errors
        error_L2 = fe.errornorm(u, u_exact, "L2")
        error_Linf = compute_vertex_linf(mesh, u, u_exact)

        print(error_L2)
        print(error_Linf)

        plot_final(u, fe.interpolate(f, V), a, "output/test/advection_diffusion/mms_multistep/final.png")
