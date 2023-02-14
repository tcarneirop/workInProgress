module sommers_mcore_search{

    use sommers_partial_search;
    use sommers_subproblem_module;
    use sommers_subproblem_explorer;
    use CTypes;
    use DynamicIters;
    use Time;

    proc sommers_call_mcore_search(const board_size:int, const initial_depth:int){

        var num_sols_search: uint(64) = 0;

        var initial_tree_size: uint(64) = 0;
        var final_tree_size: uint(64) = 0;
        var pool_size: uint(64) = 0;
        var metrics: (uint(64),uint(64)) =  (0,0);
       

        //let's change this
        var subproblem_pool: [0..#99999] Sommers_subproblem;

        metrics += queens_sommers_partial_search(board_size,initial_depth, subproblem_pool);
        pool_size = metrics[1];
        metrics[1] = 0;

        var rangeDynamic: range = 0..#pool_size;

        forall subproblem in dynamic(rangeDynamic) with (+reduce metrics) do{
        
        //forall subproblem in 0..#pool_size with (+reduce metrics) do{
           metrics += queens_sommers_final_search(board_size, initial_depth, subproblem_pool[subproblem]);
        }
        
        writeln(metrics);

        //  select scheduler{

        //     when "static" {
        //         forall idx in 0..initial_num_prefixes-1 with (+ reduce metrics) do {
        //              metrics+=queens_node_subtree_exporer(size,initial_depth,set_of_nodes[idx].board,set_of_nodes[idx].control);    
        //         }
        //     }
        //     when "dynamic" {
        //         forall idx in dynamic(rangeDynamic, chunk, num_threads) with (+ reduce metrics) do {
        //             metrics+=queens_node_subtree_exporer(size,initial_depth,set_of_nodes[idx:uint(64)].board,set_of_nodes[idx:uint(64)].control);
        //         }
        //     }
        //     when "guided" {
        //         forall idx in guided(rangeDynamic,num_threads) with (+ reduce metrics) do {
        //             metrics+=queens_node_subtree_exporer(size,initial_depth,set_of_nodes[idx:uint(64)].board, set_of_nodes[idx:uint(64)].control);
        //         }
        //     }
        //     when "stealing" {
        //         forall idx in adaptive(rangeDynamic,num_threads) with (+ reduce metrics) do {
        //             metrics+=queens_node_subtree_exporer(size,initial_depth,set_of_nodes[idx:uint(64)].board, set_of_nodes[idx:uint(64)].control);
        //         }
        //     }
        //     otherwise{
        //         writeln("\n\n ###### error ######\n\n ###### error ######\n\n ###### error ###### ");
        //     }
        // }//select

    }




    

}//module