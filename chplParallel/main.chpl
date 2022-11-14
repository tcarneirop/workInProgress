use CTypes;
use Time;
use sommers_serial;

config const size: int = 12;
config const initial_depth: int = 2;
config const second_depth:  c_int = 7;


queens_sommers64(size);

