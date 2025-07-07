"""
Analytic toy problem with quadratic potentials.
"""
module Dipole3d

import ElectromagneticFields.Dipole

export initial_conditions_dipole

export hamiltonian

Dipole.@code() # inject magnetic field code

const Δt = 0.03
const tspan = (0.0, 10.0)
# const tspan = (0.0, 999.99)

initial_conditions_dipole() = (q=[1.0, 2.0, 1.0], p=[136.07866682128767, -68.04957507731051, 0.00409666666666832], params=(μ=1E-2,))

include("guiding_center_3d_equations.jl")
include("guiding_center_3d_diagnostics.jl")

end
