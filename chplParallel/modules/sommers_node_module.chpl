

module sommers_node_module{

     use CTypes;

     require "headers/GPU_sommers.h";
     const MAX_BOARDSIZE: int = 64;

     extern "SommersSubproblem" record Sommers_subproblem{

          var  aQueenBitRes: c_array(c_longlong, 64); /* results */
          var  aQueenBitCol:  c_array(c_longlong, 64); /* marks columns which already have queens */
          var  aQueenBitPosDiag: c_array(c_longlong, 64); /* marks "positive diagonals" which already have queens */
          var  aQueenBitNegDiag: c_array(c_longlong, 64); /* marks "negative diagonals" which already have queens */
          var  subproblem_stack: c_array(c_longlong, 64+2); /* we use a stack instead of recursion */

          var pnStackPos: c_longlong;
          var numrows: c_longlong;
          var num_sols_sub: c_ulonglong;
     };


}//end of module
