

module sommers_node_module{

     use CTypes;

     require "headers/GPU_sommers.h";


     extern "SommersSubproblem" record sommers_subproblem{

          var  aQueenBitRes[MAX_BOARDSIZE]; /* results */
          var  aQueenBitCol[MAX_BOARDSIZE]; /* marks columns which already have queens */
          var  aQueenBitPosDiag[MAX_BOARDSIZE]; /* marks "positive diagonals" which already have queens */
          var  aQueenBitNegDiag[MAX_BOARDSIZE]; /* marks "negative diagonals" which already have queens */
          var  subproblem_stack[MAX_BOARDSIZE+2]; /* we use a stack instead of recursion */

          var pnStackPos: c_longlong;
          var numrows: c_longlong;
          var num_sols_sub: c_ulonglong;
     };


}//end of module
