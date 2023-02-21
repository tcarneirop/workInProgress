#ifndef GPU_QUEENS_H
#define GPU_QUEENS_H

#define MAX_BOARDSIZE 32


typedef struct sommers_subproblem{
    
    long long  aQueenBitRes[MAX_BOARDSIZE]; /* results */
    long long  aQueenBitCol[MAX_BOARDSIZE]; /* marks columns which already have queens */
    long long  aQueenBitPosDiag[MAX_BOARDSIZE]; /* marks "positive diagonals" which already have queens */
    long long  aQueenBitNegDiag[MAX_BOARDSIZE]; /* marks "negative diagonals" which already have queens */
    long long  subproblem_stack[MAX_BOARDSIZE+2]; /* we use a stack instead of recursion */
    long long   pnStackPos;

} SommersSubproblem;



#ifdef __cplusplus
extern "C" {
#endif

void GPU_sommers_call_multigpu_kernel(long long board_size, long long cutoff_depth, int block_size, 
    unsigned long long n_explorers,  SommersSubproblem *subproblems_h, 
    unsigned long long *vector_of_tree_size_h, unsigned long long *sols_h, int gpu_id);

#ifdef __cplusplus
}
#endif



#endif
