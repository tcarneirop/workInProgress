
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cmath>
#include <omp.h>
#include <sys/time.h>

#define MAX_BOARDSIZE 32

#define __BLOCK_SIZE__ 32

typedef unsigned long long SOLUTIONTYPE;

#define MIN_BOARDSIZE 2

SOLUTIONTYPE g_numsolutions = 0ULL;



double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void displayBitsLLU(unsigned long long value){

    unsigned long long  SHIFT = 8 * sizeof( unsigned long long ) - 1;
    unsigned long long MASK = (unsigned long long)1 << SHIFT ;// joga do menos para o mais significativo

    //cout<< setw(7) << value << " = ";

    for (unsigned long long c = 1; c <= SHIFT +1; c++){
        printf("%llu", value & MASK ? 1LLU: 0LLU);
        value <<= 1;
        if (c % 8 == 0)
            printf(" ");

    }

    printf("\n");

}


typedef struct subproblem{
    
    long long  aQueenBitRes[MAX_BOARDSIZE]; /* results */
    long long  aQueenBitCol[MAX_BOARDSIZE]; /* marks colummns which already have queens */
    long long  aQueenBitPosDiag[MAX_BOARDSIZE]; /* marks "positive diagonals" which already have queens */
    long long  aQueenBitNegDiag[MAX_BOARDSIZE]; /* marks "negative diagonals" which already have queens */
    long long  subproblem_stack[MAX_BOARDSIZE+2]; /* we use a stack instead of recursion */
    long long   pnStackPos;
    unsigned long long num_sols_sub;
} Subproblem;


unsigned long long partial_search_64(long long board_size, long long cutoff_depth, Subproblem *subproblem_pool)
{

    long long aQueenBitRes[MAX_BOARDSIZE]; /* results */
    long long aQueenBitCol[MAX_BOARDSIZE]; /* marks colummns which already have queens */
    long long aQueenBitPosDiag[MAX_BOARDSIZE]; /* marks "positive diagonals" which already have queens */
    long long aQueenBitNegDiag[MAX_BOARDSIZE]; /* marks "negative diagonals" which already have queens */
    long long aStack[MAX_BOARDSIZE + 2]; /* we use a stack instead of recursion */
 
    
    register long long int *pnStack;

    register long long int pnStackPos = 0LLU;

    register long long numrows = 0LL; /* numrows redundant - could use stack */
    register unsigned long long lsb; /* least significant bit */
    register unsigned long long bitfield; /* bits which are set mark possible positions for a queen */
    long long i;
    long long odd = board_size & 1LL; /* 0 if board_size even, 1 if odd */
    
    //Change here for the pool
    //long long int board_minus = 45LL; /* board size - 1 */
    long long mask = (1LL << board_size) - 1LL; /* if board size is N, mask consists of N 1's */

    unsigned long long tree_size = 0ULL;
    /* Initialize stack */
    aStack[0] = -1LL; /* set sentinel -- signifies end of stack */

    /* NOTE: (board_size & 1) is true iff board_size is odd */
    /* We need to loop through 2x if board_size is odd */
    for (i = 0; i < (1 + odd); ++i)
    {
        /* We don't have to optimize this part; it ain't the
           critical loop */
        bitfield = 0ULL;
        if (0LL == i)
        {
            /* Handle half of the board, except the middle
               column. So if the board is 5 x 5, the first
               row will be: 00011, since we're not worrying
               about placing a queen in the center column (yet).
            */
            long long int half = board_size>>1LL; /* divide by two */
            /* fill in rightmost 1's in bitfield for half of board_size
               If board_size is 7, half of that is 3 (we're discarding the remainder)
               and bitfield will be set to 111 in binary. */
            bitfield = (1LL << half) - 1LL;
            pnStack = aStack + 1LL; /* stack pointer */
            
            pnStackPos++;

            aQueenBitRes[0] = 0LL;
            aQueenBitCol[0] = aQueenBitPosDiag[0] = aQueenBitNegDiag[0] = 0LL;

        }
        else
        {
            /* Handle the middle column (of a odd-sized board).
               Set middle column bit to 1, then set
               half of next row.
               So we're processing first row (one element) & half of next.
               So if the board is 5 x 5, the first row will be: 00100, and
               the next row will be 00011.
            */
            bitfield = 1 << (board_size >> 1);
            numrows = 1; /* prob. already 0 */

            /* The first row just has one queen (in the middle column).*/
            aQueenBitRes[0] = bitfield;
            aQueenBitCol[0] = aQueenBitPosDiag[0] = aQueenBitNegDiag[0] = 0LL;
            aQueenBitCol[1] = bitfield;

            /* Now do the next row.  Only set bits in half of it, because we'll
               flip the results over the "Y-axis".  */
            aQueenBitNegDiag[1] = (bitfield >> 1ULL);
            aQueenBitPosDiag[1] = (bitfield << 1ULL);
            pnStack = aStack + 1LL; /* stack pointer */
            
            pnStackPos++;

            *pnStack++ = 0LL; /* we're done w/ this row -- only 1 element & we've done it */
            bitfield = (bitfield - 1ULL) >> 1ULL; /* bitfield -1 is all 1's to the left of the single 1 */
        }

        /* this is the critical loop */
        for (;;)
        {
         
            lsb = -((signed long long)bitfield) & bitfield; /* this assumes a 2's complement architecture */
            
            if (0ULL == bitfield)
            {
                
                bitfield = *--pnStack; /* get prev. bitfield from stack */
                pnStackPos--;

                if (pnStack == aStack) { /* if sentinel hit.... */
                    break ;
                }
                
                --numrows;
                continue;
            }
            
            bitfield &= ~lsb; /* toggle off this bit so we don't try it again */
            aQueenBitRes[numrows] = lsb; /* save the result */

            if (numrows < cutoff_depth) /* we still have more rows to process? */
            {
                long long n = numrows++;
                aQueenBitCol[numrows] = aQueenBitCol[n] | lsb;
                aQueenBitNegDiag[numrows] = (aQueenBitNegDiag[n] | lsb) >> 1LL;
                aQueenBitPosDiag[numrows] = (aQueenBitPosDiag[n] | lsb) << 1LL;

                pnStackPos++;
                
                *pnStack++ = bitfield;
                
                bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
                
                ++tree_size;

                if(numrows == cutoff_depth){

                   // printf("\nSub: ");
                   // for(int i = 0; i<cutoff_depth;++i){
                   //      printf(" %d - ", (unsigned int)(log2(aQueenBitRes[i])+1));
                   // }
                    
                   // printf("\n");
                 
                    memcpy(subproblem_pool[g_numsolutions].aQueenBitRes, aQueenBitRes, sizeof(long long)*MAX_BOARDSIZE);
                    memcpy(subproblem_pool[g_numsolutions].aQueenBitCol, aQueenBitCol, sizeof(long long)*MAX_BOARDSIZE);
                    memcpy(subproblem_pool[g_numsolutions].aQueenBitPosDiag, aQueenBitPosDiag, sizeof(long long)*MAX_BOARDSIZE);
                    memcpy(subproblem_pool[g_numsolutions].aQueenBitNegDiag, aQueenBitNegDiag, sizeof(long long)*MAX_BOARDSIZE);
                    memcpy(subproblem_pool[g_numsolutions].subproblem_stack, aStack, sizeof(long long)*(MAX_BOARDSIZE+2));
                   
                    ++g_numsolutions;

                } //if partial solution

                continue;
            }
            else
            {
                  
                bitfield = *--pnStack;
                pnStackPos--;
                --numrows;   
                continue;
            }
        }
    }

    return tree_size;

}



__global__ void gpu_final_search_64(long long board_size, long long cutoff_depth, 
    unsigned long long num_subproblems, Subproblem* __restrict__ subproblems, 
    unsigned long long * __restrict__ tree_size_d,
    unsigned long long * __restrict__ sols_d){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_subproblems) {


        //Bring this to the thread stack
       // long long* aQueenBitRes =     subproblems[idx].aQueenBitRes; 
       // long long* aQueenBitCol =     subproblems[idx].aQueenBitCol; 
       // long long* aQueenBitPosDiag = subproblems[idx].aQueenBitPosDiag; 
       // long long* aQueenBitNegDiag = subproblems[idx].aQueenBitNegDiag ; 
       // long long* aStack =           subproblems[idx].subproblem_stack ; 

        long long aQueenBitCol[MAX_BOARDSIZE]; 
        long long aQueenBitPosDiag[MAX_BOARDSIZE]; 
        long long aQueenBitNegDiag[MAX_BOARDSIZE];  
        long long aStack[MAX_BOARDSIZE]; 

        //long long int pnStackPos = subproblems[idx].pnStackPos;

        long long *pnStack;

        long long int board_minus = board_size - 1LL; /* board size - 1 */
        long long int mask = (1LL << board_size) - 1LL; /* if board size is N, mask consists of N 1's */

        unsigned long long local_num_sols = 0ULL;

        unsigned long long lsb; 
        unsigned long long bitfield; 

        unsigned long long tree_size = 0ULL;

        long long numrows = cutoff_depth;


        for(int i = 0; i<cutoff_depth+2;++i){
            aQueenBitCol[i]  =     subproblems[idx].aQueenBitCol[i] ; 
            aQueenBitPosDiag[i]  = subproblems[idx].aQueenBitPosDiag[i] ; 
            aQueenBitNegDiag[i]  = subproblems[idx].aQueenBitNegDiag[i]  ; 
            aStack[i]  =            subproblems[idx].subproblem_stack[i]  ; 
        }
        

        pnStack = aStack; /* stack pointer */
        bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
                
        for (;;)
        {
    
            lsb = -((signed long long)bitfield) & bitfield; /* this assumes a 2's complement architecture */
            
            if (0ULL == bitfield)
            {
               
                if(numrows <= cutoff_depth){ 
                    //printf("\nEND OF THE SUBPROBLEM EXPLORATION! %d", numrows);
                    break ;
                }

                bitfield = *--pnStack; /* get prev. bitfield from stack */
                
                //printf("Backtracking!");
                --numrows;
                continue;
            }

            bitfield &= ~lsb; /* toggle off this bit so we don't try it again */

            if (numrows < board_minus) /* we still have more rows to process? */
            {
            
                long long n = numrows++;
                aQueenBitCol[numrows] = aQueenBitCol[n] | lsb;
                aQueenBitNegDiag[numrows] = (aQueenBitNegDiag[n] | lsb) >> 1LL;
                aQueenBitPosDiag[numrows] = (aQueenBitPosDiag[n] | lsb) << 1LL;
                *pnStack++ = bitfield;

                bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
                ++tree_size;
                continue;
            }
            else
            {

                ++local_num_sols;
                bitfield = *--pnStack;
                --numrows;
                continue;
            }
        }

        //returning the number of solutions
        sols_d[idx] = local_num_sols;
        tree_size_d[idx] = tree_size;


    }//num_subproblems

}//kernel


unsigned long long mcore_final_search(long long board_size, long long cutoff_depth, Subproblem* subproblem, int index)
{

    long long* aQueenBitRes = subproblem->aQueenBitRes; 
    long long* aQueenBitCol = subproblem->aQueenBitCol; 
    long long* aQueenBitPosDiag = subproblem->aQueenBitPosDiag; 
    long long* aQueenBitNegDiag = subproblem->aQueenBitNegDiag ; 
    long long* aStack = subproblem->subproblem_stack ; 

    register long long *pnStack;
    
    //long long int pnStackPos = subproblem->pnStackPos;

    long long int board_minus = board_size - 1LL; /* board size - 1 */
    long long int mask = (1LL << board_size) - 1LL; /* if board size is N, mask consists of N 1's */

    unsigned long long local_num_sols = 0ULL;
    unsigned long long tree_size = 0ULL;
    
    register unsigned long long lsb; 
    register unsigned long long bitfield; 

    
    register long long numrows = cutoff_depth;

    //pnStack = aStack + pnStackPos; /* stack pointer */
    pnStack = aStack; /* stack pointer */
    
    bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
    
    /* this is the critical loop */
    for (;;)
    {
    
        lsb = -((signed long long)bitfield) & bitfield; /* this assumes a 2's complement architecture */
        
        if (0ULL == bitfield)
        {
           
            if(numrows <= cutoff_depth){ 
                //printf("\nEND OF THE SUBPROBLEM EXPLORATION! %d", numrows);
                break ;
            }

            bitfield = *--pnStack; /* get prev. bitfield from stack */
            
            //printf("Backtracking!");
            --numrows;
            continue;
        }

        bitfield &= ~lsb; /* toggle off this bit so we don't try it again */

        aQueenBitRes[numrows] = lsb; /* save the result */
        
        if (numrows < board_minus) /* we still have more rows to process? */
        {
        
            long long n = numrows++;
            aQueenBitCol[numrows] = aQueenBitCol[n] | lsb;
            aQueenBitNegDiag[numrows] = (aQueenBitNegDiag[n] | lsb) >> 1LL;
            aQueenBitPosDiag[numrows] = (aQueenBitPosDiag[n] | lsb) << 1LL;
            *pnStack++ = bitfield;

            bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
            ++tree_size;
            continue;
        }
        else
        {

            ++local_num_sols;
            bitfield = *--pnStack;
            --numrows;
            continue;
        }
    }

    //returning the number of solutions
    subproblem->num_sols_sub = local_num_sols;

    return tree_size;
}



unsigned long long call_mcore_final_search(long long board_size, unsigned long long num_subproblems, long long cutoff_depth, int chunk, 
    Subproblem *subproblem_pool, unsigned long long *num_sols){

    unsigned long long mcore_tree_size[omp_get_num_procs()];
    unsigned long long num_sols_search = 0ULL;
    unsigned long long total_mcore_tree_size = 0ULL;

    printf("\n### MCORE Search ###\n\tNumber of subproblems: %lld - Size: %lld, Initial depth: %lld, Chunk: %d, Num threads: %d\n", num_subproblems, board_size, cutoff_depth, 8, omp_get_num_procs());
    
    for(int i = 0; i<omp_get_num_procs();++i)
        mcore_tree_size[i] = 0ULL;

    #pragma omp parallel for schedule(dynamic,8) default(none)\
    shared(num_subproblems,board_size, cutoff_depth, subproblem_pool)\
    reduction(+:mcore_tree_size,num_sols_search)
    for(int s = 0; s<num_subproblems; ++s){
        mcore_tree_size[omp_get_thread_num()]+=mcore_final_search(board_size, cutoff_depth, subproblem_pool+s, s);
        num_sols_search+=subproblem_pool[s].num_sols_sub;
    }

    printf("\nTree for each thread: ");
    for(int i = 0; i<omp_get_num_procs();++i){
        printf("\nThread %d: %llu", i, mcore_tree_size[i]);
        total_mcore_tree_size+=mcore_tree_size[i];
    }
    
    *num_sols = num_sols_search;

    return total_mcore_tree_size;
}



void call_mcore_search(long long board_size, long long cutoff_depth, int chunk){
    
    unsigned long long num_sols_search = 0ULL;
    unsigned long long num_subproblems = 0ULL;
    double initial_time = rtclock();

    Subproblem *subproblem_pool = (Subproblem*)(malloc(sizeof(Subproblem)*(unsigned)10000000));

    unsigned long long initial_tree_size = partial_search_64(board_size,cutoff_depth, subproblem_pool);
    unsigned long long total_mcore_tree_size = 0ULL;
    num_subproblems = g_numsolutions;
    g_numsolutions = 0ULL;

    printf("\nPartial tree: %llu -- Pool: %llu \n", initial_tree_size, num_subproblems);


    total_mcore_tree_size =+call_mcore_final_search(board_size, num_subproblems, cutoff_depth, chunk, subproblem_pool, &num_sols_search);

    if (num_sols_search != 0)
    {
        printf("\nPARALLEL SEARCH: size %lld, Tree: %llu,  solutions: %llu\n", board_size,total_mcore_tree_size+initial_tree_size, num_sols_search*2);
    }
    else
    {
        printf("No solutions found.\n");
    }
    printf("\n#######################################\n");

    double final_time = rtclock();
    printf("\nElapsed total: %.3f\n", (final_time-initial_time));

    free(subproblem_pool);

}////////////////////////////////////////////////


void call_multigpu_kernel(long long board_size, long long cutoff_depth, int block_size, 
    unsigned long long n_explorers,  Subproblem *subproblems_h, unsigned long long *vector_of_tree_size_h, unsigned long long *sols_h, int gpu_id){

    unsigned long long int *vector_of_tree_size_d;
    unsigned long long int *sols_d;
    Subproblem *subproblems_d;

    cudaSetDevice(gpu_id);

    cudaFuncSetCacheConfig(gpu_final_search_64, cudaFuncCachePreferL1);

    cudaMalloc((void**) &vector_of_tree_size_d,n_explorers*sizeof(unsigned long long int));
    cudaMalloc((void**) &sols_d,n_explorers*sizeof(unsigned long long int));
    cudaMalloc((void**) &subproblems_d,n_explorers*sizeof(Subproblem));

    cudaMemcpy(subproblems_d, subproblems_h, n_explorers * sizeof(Subproblem), cudaMemcpyHostToDevice);

    int num_blocks = ceil((double)n_explorers/ block_size);

    printf("\n\n### 64 bits-based Kernel on device %d ###\n", gpu_id);
    printf("\nSubproblems: %lld - Num blocks: %d - Block size: %d ###\n", n_explorers, num_blocks, block_size);

    gpu_final_search_64<<< num_blocks, block_size>>>(board_size,cutoff_depth,n_explorers,subproblems_d,vector_of_tree_size_d,sols_d);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cudaMemcpy(vector_of_tree_size_h,vector_of_tree_size_d,n_explorers*sizeof(unsigned long long int),cudaMemcpyDeviceToHost);
    cudaMemcpy(sols_h,sols_d,n_explorers*sizeof(unsigned long long int),cudaMemcpyDeviceToHost);

    cudaFree(vector_of_tree_size_d);
    cudaFree(sols_d);
    cudaFree(subproblems_d);


}


void call_gpu_search_64(long long board_size, long long cutoff_depth, int block_size){

    printf("\n### 64 bits-based BP-DFS search. ###\n");

    double initial_time = rtclock();
    unsigned long long num_sols_search = 0ULL;
    unsigned long long n_explorers = 0ULL;
    Subproblem *subproblems_h = (Subproblem*)(malloc(sizeof(Subproblem)* (unsigned)10000000));

    unsigned long long gpu_tree_size = 0ULL;
    unsigned long long initial_tree_size = partial_search_64(board_size,cutoff_depth, subproblems_h);
    n_explorers = g_numsolutions;

    printf("\nPartial serach: %llu -- Pool: %llu \n", initial_tree_size, n_explorers);

    unsigned long long int *vector_of_tree_size_h = (unsigned long long int*)malloc(sizeof(unsigned long long int)*n_explorers );
    unsigned long long int *sols_h = (unsigned long long int*)malloc(sizeof(unsigned long long int)*n_explorers );

    call_multigpu_kernel(board_size, cutoff_depth,block_size, n_explorers, subproblems_h, vector_of_tree_size_h, sols_h, 0);

    for(int i = 0; i<n_explorers;++i){
        if(sols_h[i]>0)
            num_sols_search += sols_h[i];
        if(vector_of_tree_size_h[i]>0)
            gpu_tree_size +=vector_of_tree_size_h[i];
    }

    double final_time = rtclock();

    printf("\nGPU Tree size: %llu\nTotal tree size: %llu\nNumber of solutions found: %llu.\n", gpu_tree_size,(initial_tree_size+gpu_tree_size),num_sols_search*2LLU );
    printf("\nElapsed total: %.3f\n", (final_time-initial_time));

    free(sols_h);
    free(vector_of_tree_size_h);

}////////////////////////////////////////////////


unsigned long long get_cpu_load(float percent, unsigned long long num_subproblems){
    return (unsigned long long)(num_subproblems*percent);
}

unsigned long long get_remainder_load(unsigned int gpu_count, unsigned long long num_subproblems){
    return num_subproblems/gpu_count;
}

void  get_load_each_gpu(unsigned long long gpu_load, int num_gpus, unsigned long long *device_load){

    for(int device = 0; device<num_gpus;++device){
        device_load[device] = gpu_load/num_gpus;
        if(device == (num_gpus-1)){
            device_load[device]+= gpu_load%num_gpus;
        }
    }
}//////////


void call_cpu_multi_gpu_search(long long board_size, long long cutoff_depth, int block_size, float cpup, int chunk){


    printf("\n### CPU-Multi-GPU bit-based BP-DFS search. ###\n");

    double initial_time = rtclock();


    unsigned long long num_sols_search = 0ULL;


    unsigned long long num_subproblems = 0ULL;
    unsigned long long gpu_load = 0ULL;
    unsigned long long cpu_load = 0ULL;

    int num_gpus = 0;
    cudaGetDeviceCount( &num_gpus );
    printf("\nNumber of GPUS: %d\n", num_gpus );

    unsigned long long device_load[num_gpus];

    Subproblem *subproblems_h = (Subproblem*)(malloc(sizeof(Subproblem)* (unsigned)10000000));


    unsigned long long gpu_tree_size = 0ULL;
    unsigned long long mcore_tree_size = 0ULL;
    unsigned long long initial_tree_size = partial_search_64(board_size,cutoff_depth, subproblems_h);

    num_subproblems = g_numsolutions;

    printf("\nPartial search:\n\tPartial tree size:%llu -- Pool: %llu \n", initial_tree_size, num_subproblems);

    unsigned long long int *vector_of_tree_size_h = (unsigned long long*)malloc(sizeof(unsigned long long)*num_subproblems);
    unsigned long long int *sols_h = (unsigned long long*)malloc(sizeof(unsigned long long)*num_subproblems);


    omp_set_nested (1);

    cpu_load = get_cpu_load(cpup, num_subproblems);
    gpu_load = num_subproblems - cpu_load;
    get_load_each_gpu(gpu_load, num_gpus, device_load);


    printf("\nTotal CPU load: %llu - CPU percent: %f - GPU load: %llu", cpu_load, cpup, gpu_load);
    printf("\nLoad of each GPU:");
    for(int device = 0; device<num_gpus;++device){
        printf("\n\tDevice: %d - load : %llu ", device, device_load[device]);

    }
    printf("\n\n");
    //exit(1);


    #pragma omp parallel for default(none)\
    shared(num_subproblems,board_size, num_gpus, cpu_load, gpu_load, block_size, cutoff_depth, device_load, subproblems_h, mcore_tree_size, vector_of_tree_size_h, sols_h,num_sols_search,chunk)
    for(int device = 0; device<=num_gpus; ++device){
        if(device<num_gpus){//gpu
            unsigned long long local_stride = device * (gpu_load/num_gpus);

            call_multigpu_kernel(board_size, cutoff_depth,block_size, device_load[device], 
                subproblems_h+local_stride, vector_of_tree_size_h+local_stride, sols_h+local_stride, device);

        }//for each gpu device
        else{
            if(cpu_load>0){
                // subproblems_h+gpu_load a stride -- after the gpu load
                //
                mcore_tree_size+=call_mcore_final_search(board_size, cpu_load, cutoff_depth, 
                    chunk, subproblems_h+gpu_load, &num_sols_search);

            }//if the GPU participates in the search
        }//cpu
       
    }
   
   
    //reduction of the GPU tree and GPU num sols
    for(int i = 0; i<gpu_load;++i){
        if(sols_h[i]>0)
            num_sols_search += sols_h[i];
        if(vector_of_tree_size_h[i]>0)
            gpu_tree_size +=vector_of_tree_size_h[i];
    }

    double final_time = rtclock();
  
    printf("\n\nTOTAL Tree size: %llu", initial_tree_size+gpu_tree_size+mcore_tree_size);
    printf("\n\tINITIAL tree size: %llu",initial_tree_size);
    if(cpu_load>0.f)
        printf("\n\tMCORE Tree size: %llu",mcore_tree_size);
    printf("\n\tGPU Tree size: %llu",gpu_tree_size);

    printf("\nNumber of solutions found: %llu", num_sols_search*2LLU );

    printf("\n\nElapsed TIME: %.3f\n", (final_time-initial_time));
    free(sols_h);
    free(vector_of_tree_size_h);

}   



/* main routine for N Queens program.*/
int main(int argc, char** argv)
{
    
    int search = atoi(argv[1]);

    int boardsize = atoi(argv[2]);

    if(search < 0 || search > 3 || argc < 4){
        printf("### Wrong Parameters ###\n");
        return 1;

    }

    if(search == 0){
        //exec, size, search, depth, chunk; 
        call_mcore_search(boardsize, (long long)(atoi(argv[3])), atoi(argv[4]));
    }
    if(search == 1){

        //I was verifying whether avoiding using 64 bits would improve something... 
        //exec, size, search, depth, block size; 
         call_gpu_search_64(boardsize, (long)(atoi(argv[3])), atoi(argv[4]));         
    }

    if(search == 2){

        //I was verifying whether avoiding using 64 bits would improve something... 
        //exec, size, search, depth, block size, cpu percent; 
        float cpup =  (atoi(argv[5]))/100.f;
        call_cpu_multi_gpu_search(boardsize, (long)(atoi(argv[3])), atoi(argv[4]), cpup, 128);         
    }
    if(search == 4){

        //I was verifying whether avoiding using 64 bits would improve something... 
        //exec, size, search, depth, block size; 
        // call_cpu_multi_gpu_search(boardsize, (long)(atoi(argv[3])), atoi(argv[4]));         
    }
    return 0;
}





void single_gpu_call_gpu_search_64(long long board_size, long long cutoff_depth, int block_size){

    printf("\n### 64 bits-based BP-DFS search. ###\n");

    double initial_time = rtclock();
    unsigned long long num_sols_search = 0ULL;
    unsigned long long n_explorers = 0ULL;
    Subproblem *subproblems_h = (Subproblem*)(malloc(sizeof(Subproblem)* (unsigned)10000000));

    unsigned long long gpu_tree_size = 0ULL;
    unsigned long long initial_tree_size = partial_search_64(board_size,cutoff_depth, subproblems_h);
    n_explorers = g_numsolutions;
   
    cudaFuncSetCacheConfig(gpu_final_search_64, cudaFuncCachePreferL1);
    
    printf("\nPartial serach: %llu -- Pool: %llu \n", initial_tree_size, n_explorers);

    unsigned long long int *vector_of_tree_size_h = (unsigned long long int*)malloc(sizeof(unsigned long long int)*n_explorers );
    unsigned long long int *sols_h = (unsigned long long int*)malloc(sizeof(unsigned long long int)*n_explorers );

    unsigned long long int *vector_of_tree_size_d;
    unsigned long long int *sols_d;
    Subproblem *subproblems_d;

    cudaMalloc((void**) &vector_of_tree_size_d,n_explorers*sizeof(unsigned long long int));
    cudaMalloc((void**) &sols_d,n_explorers*sizeof(unsigned long long int));
    cudaMalloc((void**) &subproblems_d,n_explorers*sizeof(Subproblem));

    cudaMemcpy(subproblems_d, subproblems_h, n_explorers * sizeof(Subproblem), cudaMemcpyHostToDevice);


    int num_blocks = ceil((double)n_explorers/ block_size);

    printf("\n### 64 bits-based Kernel ###\n");

    gpu_final_search_64<<< num_blocks, block_size>>>(board_size,cutoff_depth,n_explorers,subproblems_d,vector_of_tree_size_d,sols_d);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    cudaMemcpy(vector_of_tree_size_h,vector_of_tree_size_d,n_explorers*sizeof(unsigned long long int),cudaMemcpyDeviceToHost);
    cudaMemcpy(sols_h,sols_d,n_explorers*sizeof(unsigned long long int),cudaMemcpyDeviceToHost);

    cudaFree(vector_of_tree_size_d);
    cudaFree(sols_d);
    cudaFree(subproblems_d);


    double final_time = rtclock();
    for(int i = 0; i<n_explorers;++i){
        if(sols_h[i]>0)
            num_sols_search += sols_h[i];
        if(vector_of_tree_size_h[i]>0)
            gpu_tree_size +=vector_of_tree_size_h[i];
    }


    printf("\nGPU Tree size: %llu\nTotal tree size: %llu\nNumber of solutions found: %llu.\n", gpu_tree_size,(initial_tree_size+gpu_tree_size),num_sols_search*2LLU );
    printf("\nElapsed total: %.3f\n", (final_time-initial_time));


    free(sols_h);
    free(vector_of_tree_size_h);
}////////////////////////////////////////////////

