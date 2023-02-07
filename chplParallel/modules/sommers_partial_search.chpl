
module sommers_partial_search{

    use sommers_node_module;
    use CTypes;

    proc queens_sommers_partial_search(const board_size:int, const initial_depth:int, 
        ref subproblems_pool: [] Sommers_subproblem){

    	var aQueenBitRes: [0..#MAX_BOARDSIZE] int;     // results 
        var aQueenBitCol: [0..#MAX_BOARDSIZE] int;     // marks colummns which already have queens 
        var aQueenBitPosDiag: [0..#MAX_BOARDSIZE] int; // marks "positive diagonals" which already have queens 
        var aQueenBitNegDiag: [0..#MAX_BOARDSIZE] int; // marks "negative diagonals" which already have queens 
        var aStack: [0..MAX_BOARDSIZE + 1] int;        // we use a stack instead of recursion 

        var stack_position: int = 0;
        var numrows: int = initial_depth;
        var lsb: uint(64);
        var bitfield: uint(64);
        var numsolutions: int = 0;
        var tree_size: uint(64) = 0;

        var i: int;
        var odd: int = board_size & 1;
        
        var mask = (1 << board_size) - 1;

        aStack[0] = -1; // set sentinel -- signifies end of stack */

        // We need to loop through 2x if board_size is odd */
        for i in 0..#(1+odd) do
        {
            
            bitfield = 0;
            
            if (0 == i) then {
                
                var half: int = (board_size>>1); // divide by two */
                
                bitfield = ((1 << half) - 1):uint(64);
                //pnStack = aStack + 1; // stack pointer */
                
                stack_position+=1;//?

                aQueenBitRes[0] = 0;
                aQueenBitCol[0] = 0;
                aQueenBitPosDiag[0] = 0;
                aQueenBitNegDiag[0] = 0;
            }
            else
            {
                bitfield = (1 << (board_size >> 1)):uint(64);
                numrows = 1; // prob. already 0 */

                // The first row just has one queen (in the middle column).*/
                aQueenBitRes[0] = bitfield:int;
                aQueenBitCol[0] = 0;
                aQueenBitPosDiag[0] = 0;
                aQueenBitNegDiag[0] = 0;
                aQueenBitCol[1] = bitfield:int;


                aQueenBitNegDiag[1] = (bitfield >> 1):int;
                aQueenBitPosDiag[1] = (bitfield << 1):int;
                
                stack_position=1;
    			
    			aStack[stack_position] = 0;
    			stack_position+=1;
  
                bitfield = (bitfield - 1) >> 1; // bitfield -1 is all 1's to the left of the single 1 */
            }//for

            // this is the critical loop */

            while(true){
            	
                lsb = -(bitfield:int) & bitfield; /* this assumes a 2's complement architecture */
                
                if (0:uint(64) == bitfield)
                {
                    
                    stack_position-=1;
        		
        			bitfield = (aStack[stack_position]):uint(64); 

                    if (aStack[stack_position] == -1) { // if sentinel hit.... */
                        break ;
                    }
                    numrows-=1;
                    continue;
                }

                bitfield &= ~lsb; 
                aQueenBitRes[numrows] = lsb:int; // save the result */

                if (numrows < initial_depth) // we still have more rows to process? */
                {
                	//long long int n = numrows++; 
                    var n: int = numrows;
                    numrows+=1;

                    aQueenBitCol[numrows] = (aQueenBitCol[n] | lsb):int;
                    aQueenBitNegDiag[numrows] = ((aQueenBitNegDiag[n] | lsb) >> 1):int;
                    aQueenBitPosDiag[numrows] = ((aQueenBitPosDiag[n] | lsb) << 1):int;
                    
                    aStack[stack_position] = bitfield:int;
                    stack_position+=1;

                    bitfield = (mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows])):uint(64);
                    tree_size+=1;

                    if(numrows == initial_depth){
                        for i in 0..initial_depth do{
                            subproblems_pool[numsolutions].aQueenBitRes[i] = aQueenBitRes[i];
                            subproblems_pool[numsolutions].aQueenBitCol[i] = aQueenBitCol[i];
                            subproblems_pool[numsolutions].aQueenBitPosDiag[i] = aQueenBitPosDiag[i];
                            subproblems_pool[numsolutions].aQueenBitNegDiag[i] = aQueenBitNegDiag[i]; 
                        }

                        for i in 0..initial_depth+2 do 
                            subproblems_pool[numsolutions].subproblem_stack[i] = aStack[i];
                   
                        numsolutions+=1;
                    }

                    continue;
                }
                else
                {
                    stack_position-=1;
                    bitfield = (aStack[stack_position]):uint(64);
                    numrows-=1;
                    continue;
                }

            }//while
        }

        writeln("\nNumber of Solutios: ", numsolutions,".\n");
        for i in 0..#numsolutions do writeln(subproblems_pool[i]);
    }//partial search
}//module