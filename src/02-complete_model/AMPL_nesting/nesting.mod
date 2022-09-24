# Optimization Project: Biscuit Optimizer
# Roberto Basla
# Politecnico di Milano
# A.Y. 2021/2022
#
# Complete model

# SETS

# Set of cookie cutters
set CUTTERS;
# Set of number of rows for each cutter
set N_CUTTER{CUTTERS};
# Set of number of columns for each cutter
set M_CUTTER{CUTTERS};

# Number of rows of the dough bitmask
param rows;
# Number of columns of the dough bitmask
param cols;
# Set of bitmask rows
set N := 1..rows;
# Set of bitmask columns
set M := 1..cols;

# PARAMETERS

# Cutter values
param c{CUTTERS} >= 0;
# Number of rows for each cutter mask
param n_rows_cutters{CUTTERS};
# Number of columns for each cutter mask
param n_columns_cutters{CUTTERS};
# Cutter bitmasks
param cutter_masks{N, M, CUTTERS};
# Dough bitmask
param dough_mask{N, M};

# VARIABLES

# Binary variable corresponding to the cutter bitmasks' top left corner
# =1 if the bitmask is placed in the coordinate
var y{N, M, CUTTERS} binary;
# Integer variable corresponding to the number of active bitmasks overlapping
# in a given pixel
var z{N, M, CUTTERS} integer;

# OBJECTIVE FUNCTION
maximize biscuit_value: 
	sum{i in CUTTERS} c[i] * sum{n in N, m in M} y[n, m, i];

# CONSTRAINTS

# Constraint for balancing biscuit number
s.t. number{i in CUTTERS, j in CUTTERS}:
	sum{n in N, m in M} y[n, m, i] <= 3 * sum{n in N, m in M} y[n, m, j];

# Constraint enforcing that no cutter should be placed in coordinates that would result in an incomplete biscuit
s.t. outside_cutter{i in CUTTERS, n in N, m in M : n + n_rows_cutters[i] > rows or m + n_columns_cutters[i] > cols}:
	y[n, m, i] = 0;

# Constraint enforcing that y variables can be active only if the cutter bitmask falls entirely in the usable
# part of the dough mask
s.t. complete_cutter{i in CUTTERS, n in N, m in M, h in N_CUTTER[i], k in M_CUTTER[i] : 
		n + h - 1 <= rows and m + k - 1 <= cols}:
	y[n, m, i] <= dough_mask[n + h - 1, m + k - 1] + (1 - cutter_masks[h, k, i]);

# Linking of z and y variables as z = sum of ys that cover z's pixel
s.t. mask_cutter{i in CUTTERS, n in N, m in M}:
	z[n, m, i] = sum{h in N_CUTTER[i], k in M_CUTTER[i] : n - (h - 1) > 0 and m - (k - 1) > 0}
		y[n - (h - 1), m - (k - 1), i] * cutter_masks[h, k, i];

# Non-overlapping constraints
s.t. no_overlap{n in N, m in M}:
	sum{i in CUTTERS} z[n, m, i] <= 1
