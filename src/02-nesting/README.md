# Nesting model solutions
This folder contains the solutions to the complete nesting problem.

## AMPL_nesting
This subfolder contains the files used to solve the problem in AMPL with CPLEX:

### nesting.dat
Data file for the nesting model, containing bitmasks as well.

### nesting.mod
Model of the nesting problem.

### nesting.dat
Run file for the nesting problem.

### output_12h.txt
Text file containing the output of the model run for 40000s.

## gurobi_nesting.ipynb
Notebook containing the nesting problem solution using Gurobi's Python API and solver, run for 40000s.

## or-tools_nesting.ipynb
Notebook containing the nesting problem solution using Google's OR-Tools Python API and CP-SAT solver, run for 40000s.
