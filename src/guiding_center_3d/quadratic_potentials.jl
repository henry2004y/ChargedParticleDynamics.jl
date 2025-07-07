"""
Analytic toy problem with quadratic potentials.
"""
module QuadraticPotentials3d

import ElectromagneticFields.QuadraticPotentials

export initial_conditions_quadratic

export hamiltonian

QuadraticPotentials.@code() # inject magnetic field code

# const Δt = 0.5
# const tspan = (0.0, 25000.0)

const Δt = 0.035
const tspan = (0.0, 12250.)

# initial_conditions_quadratic() = (q=[0.3, 0.2, -1.4], p=[-9.999601705247228, 14.999402557870841, 0.26414737638644503], params=(μ=2.5E-3,))
initial_conditions_quadratic() = (q=[0.301, 0.207, -1.4], p=[-10.349589576227105, 15.049403200214291, 0.2649973540554351], params=(μ=2.5E-3,))
# initial_conditions_quadratic() = (q=[0.301, 0.207, -1.4], p=[-10.35, 15.05, 0.265], params=(μ=2.5E-3,))

include("guiding_center_3d_equations.jl")
include("guiding_center_3d_diagnostics.jl")

end
