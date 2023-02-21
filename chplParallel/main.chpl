use CTypes;
use Time;
use sommers_serial;
use sommers_mcore_search;
use sommers_subproblem_module;
use sommers_subproblem_explorer;
use sommers_partial_search;


config const size: int = 12;
config const initial_depth: int = 2;
config const second_depth:  c_int = 7;


//queens_sommers64(size);
sommers_call_mcore_search(size,initial_depth);
