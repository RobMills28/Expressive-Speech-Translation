"""
Helpers for distributed training.
"""
import io
import os
import socket
import pickle  # Add this import

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3

def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    print("MPI.COMM_WORLD.Get_rank()", MPI.COMM_WORLD.Get_rank())
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"
    print('os.environ["CUDA_VISIBLE_DEVICES"]', os.environ["CUDA_VISIBLE_DEVICES"])
    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    With multiple fallback mechanisms for handling pickled models.
    """
    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    # Define a custom persistent_load function that just returns the ID
    def persistent_load(pid):
        return pid

    # Try multiple loading methods in sequence with proper error handling
    try:
        # Method 1: Try with persistent_load
        if "map_location" in kwargs:
            return th.load(io.BytesIO(data), 
                          map_location=kwargs["map_location"],
                          pickle_module=pickle, 
                          persistent_load=persistent_load)
        else:
            return th.load(io.BytesIO(data),
                          pickle_module=pickle,
                          persistent_load=persistent_load)
    except Exception as e1:
        print(f"First loading attempt failed: {str(e1)}")
        try:
            # Method 2: Try with weights_only=True
            if "map_location" in kwargs:
                return th.load(io.BytesIO(data), 
                              map_location=kwargs["map_location"],
                              weights_only=True)
            else:
                return th.load(io.BytesIO(data), 
                              weights_only=True)
        except Exception as e2:
            print(f"Second loading attempt failed: {str(e2)}")
            try:
                # Method 3: Last resort, try with default parameters
                if "map_location" in kwargs:
                    return th.load(io.BytesIO(data), 
                                  map_location=kwargs["map_location"])
                else:
                    return th.load(io.BytesIO(data))
            except Exception as e3:
                print(f"All loading attempts failed.\nErrors:\n1: {str(e1)}\n2: {str(e2)}\n3: {str(e3)}")
                raise

def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)

def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()