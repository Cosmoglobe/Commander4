def identify_masters(global_comm, split_comm, split_id):
    """Identify and share which global ranks are masters of each split."""
    # Prepare data: only sub-masters (local rank 0) provide meaningful data
    data = (split_id, global_comm.Get_rank()) if split_comm.Get_rank() == 0 else None
    
    # Gather data from all processes to all processes
    all_data = global_comm.allgather(data)
    
    # Process gathered data into a dictionary mapping split_ids to master ranks
    return {item[0]: item[1] for item in all_data if item is not None}
