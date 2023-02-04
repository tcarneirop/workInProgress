use CTypes;
use Time;
use sommers_serial;
use sommers_node_module;
use sommers_partial_search;


config const size: int = 12;
config const initial_depth: int = 2;
config const second_depth:  c_int = 7;

var subproblem_pool: [0..#999999] Sommers_subproblem;


queens_sommers64(size);
queens_sommers_partial_search(size,initial_depth, subproblem_pool);
