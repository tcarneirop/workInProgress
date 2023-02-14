

module sommers_subproblem_module{

     use CTypes;

     require "headers/GPU_sommers.h";
     const MAX_BOARDSIZE: int = 32;

     extern "SommersSubproblem" record Sommers_subproblem{

          var  aQueenBitRes: c_array(c_longlong, 32); /* results */
          var  aQueenBitCol:  c_array(c_longlong, 32); /* marks columns which already have queens */
          var  aQueenBitPosDiag: c_array(c_longlong, 32); /* marks "positive diagonals" which already have queens */
          var  aQueenBitNegDiag: c_array(c_longlong, 32); /* marks "negative diagonals" which already have queens */
          var  subproblem_stack: c_array(c_longlong, 32+2); /* we use a stack instead of recursion */
     };


}//end of module
