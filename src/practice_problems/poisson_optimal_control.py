"""This is based on the Poisson optimal control example at: http://www.dolfin-adjoint.org/en/latest/documentation/poisson-mother/poisson-mother.html

This code solves a PDE-constrained optimisation problem.

The governing PDE is the BV problem for the Poisson equation:

    - kappa * grad^2(u) = f on \Sigma
    u = 0 on \partial \Sigma

Given a desired temperature profile d, we want to optimise f s.t. u is as close as possible to d. We also add Tikhonov regularisation to make the problem well-posed:

    min_f \int (u-d)^2 dx + alpha * \int f^2 dx

This code plots out various visualisations in the same directory.

"""

from dolfin import *
from dolfin_adjoint import *
import moola
import matplotlib.pyplot as plt


set_log_level(20)  # 30 = Warning, 20=Info


n = 32
mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "DG", 0)
kappa = Constant(1.0)  # Thermal diffusivity [m^2 / s]
f = interpolate(Expression("x[0]+x[1]", name='Control', degree=1), W)  # Our control variable
u = Function(V, name="Temperature")  # Our solution variable 

cb_storage = []  # Store intermediate control solutions here


def plot_forward_soln(u, f, filename):
    """Plot source term and temperature solution
    """
    plt.clf()
    fig = plt.figure(figsize=(8,4))

    ax1 = fig.add_subplot(1, 2, 1)
    c = plot(u)
    cbar = fig.colorbar(c, ax=ax1)
    cbar.set_label("[temperature]")
    ax1.set_title("Solved temperature")

    ax2 = fig.add_subplot(1, 2, 2)
    c = plot(f)
    cbar = fig.colorbar(c, ax=ax2)
    cbar.set_label("[temperature / time]")
    ax2.set_title("Source term")

    fig.tight_layout()
    plt.savefig(filename)
    print(f"Written plot to: {filename}")


def poisson_forward(mesh, u, V, W, f, kappa):
    """Forward solution of poisson problem, given source term
    """
    v = TestFunction(V)
    
    lhs = kappa * inner(grad(u), grad(v)) * dx
    rhs = f * v * dx
    F = lhs - rhs
    bc = DirichletBC(V, 0.0, "on_boundary")
    solve(F == 0, u, bc)

    return u


def get_target_temp_params():
    """Parameters used in analytic expression of target temperature
    """
    w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
    alpha = Constant(1e-6)
    d = 1 / (2 * pi ** 2)
    d = Expression("d*w", d=d, w=w, degree=3)
    return w, alpha, d



def optimiser_cb(j, dj, m):
    """Callback to retrieve the control param from each optimisation step
    """
    print("Calling ReducedFunctional derivative_cb_post callback")
    copy = m.copy()
    cb_storage.append(copy)


def define_error_functional(mesh, u, V, W, f):
    """Error functional and reduced functional definition
    """
    _, alpha, d = get_target_temp_params()
    J = assemble((0.5 * inner(u - d, u - d)) * dx + alpha / 2 * f ** 2 * dx)
    control = Control(f)
    rf = ReducedFunctional(J, control, derivative_cb_post=optimiser_cb)

    return rf


def optimise_control(rf, f):
    """Optimiser code
    """
    problem = MoolaOptimizationProblem(rf)
    f_moola = moola.DolfinPrimalVector(f)
    solver = moola.NewtonCG(problem, f_moola, options={'gtol': 1e-9,
                                                       'maxiter': 20,
                                                       'display': 3,
                                                       'ncg_hesstol': 0})
    sol = solver.solve()
    
    return sol


def plot_intermediate_source_terms(filename):
    """Plot all the intermediate source terms found at each optimisation step
    """
    cb_storage_filtered = cb_storage[::2]
    n = len(cb_storage_filtered)

    plt.clf()
    fig = plt.figure(figsize=(4*n, 4))

    for i in range(n):
        ax = fig.add_subplot(1, n, i + 1)
        c = plot(cb_storage_filtered[i])
        cb = fig.colorbar(c, ax=ax)
        cb.set_label(["temperature / time"])
        ax.set_title(f"Source term iteration {i}")

    fig.tight_layout()
    plt.savefig(filename)
    print(f"Written plot to: {filename}")



def plot_final_solution(u, f_opt, V, filename):
    """Plot the final temperature/source terms after optimisation has terminated
    """
    _, _, d = get_target_temp_params()
    d = interpolate(d, V)
    
    plt.clf()
    fig = plt.figure(figsize=(8,8))

    ax1 = fig.add_subplot(2, 2, 1)
    c = plot(u)
    cb = fig.colorbar(c, ax=ax1)
    cb.set_label("[temperature]")
    ax1.set_title("Solved temperature")

    ax2 = fig.add_subplot(2, 2, 2)
    c = plot(d)
    cb = fig.colorbar(c, ax=ax2)
    cb.set_label("[temperature]")
    ax2.set_title("Target temperature")

    ax3 = fig.add_subplot(2, 2, 3)
    c = plot(f_opt)
    cb = fig.colorbar(c, ax=ax3)
    cb.set_label("[temperature / time]")
    ax3.set_title("Source term")

    ax4 = fig.add_subplot(2, 2, 4)
    c = plot((u - d))
    cb = fig.colorbar(c, ax=ax4)
    cb.set_label("[temperature]")
    ax4.set_title("Temperature error")

    fig.tight_layout()
    plt.savefig(filename)
    print(f"Written plot to: {filename}")


def compute_solution_error(mesh, sol, f_opt):
    """Write out errors to console
    """
    # Define the expressions of the analytical solution
    w, alpha, d = get_target_temp_params()
    f_analytic = Expression("1/(1+alpha*4*pow(pi, 4))*w", w=w, alpha=alpha, degree=3)
    u_analytic = Expression("1/(2*pow(pi, 2))*f", f=f_analytic, degree=3)
    
    # We can then compute the errors between numerical and analytical solutions.
    control_error = errornorm(f_analytic, f_opt)
    state_error = errornorm(u_analytic, sol)
    print("h(min):           %e." % mesh.hmin())
    print("Error in state:   %e." % state_error)
    print("Error in control: %e." % control_error)


def main():
    sol = poisson_forward(mesh, u, V, W, f, kappa)
    plot_forward_soln(sol, f, "linear_temp_solve_IC.png")

    rf = define_error_functional(mesh, u, V, W, f)
    sol = optimise_control(rf, f)
    f_opt = sol["control"].data

    plot_intermediate_source_terms("poisson_intermediate_source_terms.png")

    f.assign(f_opt)
    sol = poisson_forward(mesh, u, V, W, f, kappa)
    plot_final_solution(sol, f_opt, V, "poisson_final_solution.png")
    compute_solution_error(mesh, sol, f_opt)

    print("Program completed normally")


if __name__ == "__main__":
    main()



