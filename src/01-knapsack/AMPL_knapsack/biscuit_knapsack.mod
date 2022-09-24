# Optimization Project: Biscuit Optimizer
# Roberto Basla
# Politecnico di Milano
# A.Y. 2021/2022
#
# Model of the integer knapsack problem

# SETS

# Number of available cutters
param n;
# Set of cutters
set I := 1..n;

# PARAMETERS

# Area of the available dough in pixels (budget)
param B >= 0;
# Value of each cutter (cost)
param c{i in I} >= 0;
# Area (in pixels) of each cutter mask (weight)
param a{i in I} >= 0;

# VARIABLES

# Integer knapsack variables
var x{i in I} >= 0, integer;

# OBJECTIVE FUNCTION
maximize biscuit_value: 
	sum{i in I} c[i] * x[i];

# CONSTRAINTS

# Knapsack budget constraint
s.t. budget:
	sum{i in I} a[i] * x[i] <= B;

# Cutter number constraints
s.t. number{i in I, j in I}:
	x[i] <= 3 * x[j];
