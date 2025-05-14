import torch
import time
import argparse
import zmq
import traceback
import threading
import logging
import secrets
import string
from typing import List
from datetime import datetime

from nixl._api import nixl_agent

def get_uuid(length=8):
    """Generate a secure random UUID of specified length."""
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def setup_logging():
    logger = logging.getLogger('benchmark')
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(f'benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    log_format = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Nixl")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the benchmark on")
    parser.add_argument("--num-blocks", type=int, default=100, help="Number of blocks to create")
    parser.add_argument("--num-layers", type=int, default=32, help="Number of layers in each block")
    parser.add_argument("--block-size", type=int, default=256, help="Size of each block")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden dimension of each block")
    parser.add_argument("--threads", type=int, default=1, help="Number of parallel threads to run")
    parser.add_argument("--iters", type=int, default=1, help="Number of transfer iters to run")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type of the blocks")

    parser.add_argument("--role", type=str, required=True, help="Role of the agent ('creator' or 'peer')")
    parser.add_argument("--operation", type=str, required=True, help="Operation to perform ('READ' or 'WRITE')")

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for ZMQ socket")
    parser.add_argument("--port", type=int, default=5555, help="Port for ZMQ socket")
    return parser.parse_args()

def init_zmq_socket(host, port, role):
    """
    Initialize the ZMQ socket for communication.
    """
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PAIR)
    if "peer" in role:
        zmq_socket.bind(f"tcp://{host}:{port}")
    else:
        zmq_socket.connect(f"tcp://{host}:{port}")
        # Ensure the socket is ready to receive messages
        zmq_socket.setsockopt(zmq.LINGER, 0)

    logger.info(f"Initialized ZMQ socket(port {port}) for role: {role}")
    return zmq_socket

def create_dataset(role,
                   device, 
                   num_blocks = 100, 
                   num_layers = 32,
                   block_size = 256,
                   hidden_dim = 1024,
                   dtype = torch.bfloat16):
    """
    Create a dataset of random tensors.
    """
    logger.info(f"Creating dataset with {num_blocks} blocks, {num_layers} layers, block_size {block_size}, hidden_dim {hidden_dim}")
    block_shape = (num_layers, 2, block_size, hidden_dim)
    dataset = []
    value = 0 if "peer" in role else 1
    for _ in range(num_blocks):
        block = torch.full(block_shape, value, device=device, dtype=dtype)
        dataset.append(block)
    logger.info(f"Dataset creation completed with {len(dataset)} blocks")
    return dataset


def create_nixl_agents(role: str, tensors: list[torch.Tensor], zmq_socket):
    """
    Create Nixl agents based on the role.
    """
    logger.info(f"Creating Nixl agent for role: {role}")
    agent = nixl_agent(role)
    register_descs = agent.register_memory(tensors)

    local_meta = agent.get_agent_metadata()

    if "creator" in role:
        zmq_socket.send(local_meta)
        remote_meta = zmq_socket.recv()
        peer_name = agent.add_remote_agent(remote_meta).decode("utf-8")
        logger.info(f"Creator added peer: {peer_name}")
        assert "peer" in peer_name, "Peer name mismatch for role=creator"
    elif "peer" in role:
        remote_meta = zmq_socket.recv()
        peer_name = agent.add_remote_agent(remote_meta).decode("utf-8")
        logger.info(f"Peer added creator: {peer_name}")
        zmq_socket.send(local_meta)
        assert "creator" in peer_name, "Peer name mismatch for role=peer"

    return agent, peer_name, register_descs

def initialize_xfer_metadata(
        role: str,
        operation: str, 
        agent: nixl_agent, 
        peer_name: str, 
        register_descs,
        zmq_socket
    ):
    """
    Initialize transfer metadata.
    """
    logger.info(f"Initializing transfer metadata for {role} with operation {operation}")
    local_xfer_descs = register_descs.trim()
    remote_xfer_descs = None
    transfer_handle = None

    if "peer" in role:
        # Wait until there is a message from the creator
        msg = zmq_socket.recv().decode("utf-8")
        if msg == "START":
            logger.info(f"{role} received START message")
        else:
            logger.error(f"{role} received unexpected message: {msg}")
            zmq_socket.close()
            exit(0)

        # send the xfer descs to the peer
        logger.info(f"{role} sending xfer descs to {peer_name}")
        zmq_socket.send(agent.get_serialized_descs(local_xfer_descs))

    elif "creator" in role:
        zmq_socket.send("START".encode("utf-8"))
        logger.info(f"{role} sent START message to {peer_name}")

        # Wait until there is a message from the peer
        msg = zmq_socket.recv()
        remote_xfer_descs = agent.deserialize_descs(msg)

        logger.info(f"{role} received xfer descs from {peer_name}")
        uid = role.split('-', 1)[-1]
        transfer_handle = agent.initialize_xfer(
                operation,
                local_xfer_descs,
                remote_xfer_descs,
                peer_name,
                uid)

    return transfer_handle

def start_transfer(
        role: str,
        agent: nixl_agent,
        transfer_handle,
        peer_name,
    ):
    logger.info(f"Starting transfer for {role}")
    if "creator" in role:
        state = agent.transfer(transfer_handle)
        if state == "ERR":
            logger.error("Error in transfer")
        while True:
            state = agent.check_xfer_state(transfer_handle)
            if state == "DONE":
                logger.info("Transfer finished in creator")
                break
            elif state == "ERR":
                logger.error("Error in transfer")
                break
    else:
        uid = peer_name.split('-', 1)[-1]
        while not agent.check_remote_xfer_done(peer_name, uid.encode("utf-8")):
            continue
        logger.info("Transfer finished in peer")



def cleanup_transfer(
        agent: nixl_agent,
        transfer_handle,
        register_descs,
    ):
    # Cleanup the transfer handle and registered descriptors
    if transfer_handle is not None:
        agent.release_xfer_handle(transfer_handle)
    agent.deregister_memory(register_descs)

def cleanup_agent(
        agent: nixl_agent,
    ):
    # Cleanup the agent
    agent.remove_remote_agent(agent.name)

def start_agent_pair(agent_name, device, op, uid):
    logger.info(f"Starting nixl agent pair {agent_name} on {device} with operation {op}")
    port_index = int(uid.split('-')[-1])
    zmq_socket = init_zmq_socket(args.host, args.port+port_index, args.role)
    
    try:
        # Create dataset
        dataset = create_dataset(agent_name, device, 
                             num_blocks=args.num_blocks, 
                             num_layers=args.num_layers,
                             block_size=args.block_size,
                             hidden_dim=args.hidden_dim,
                             dtype=getattr(torch, args.dtype))
        
        # Create Nixl agents
        agent, peer_name, register_descs = create_nixl_agents(agent_name, dataset, zmq_socket)

        # Initialize transfer metadata
        start = time.perf_counter()
        transfer_handle = initialize_xfer_metadata(
            agent_name,
            op,
            agent,
            peer_name,
            register_descs,
            zmq_socket
        )
        end = time.perf_counter()
        logger.info(f"Time to initialize transfer metadata: {end - start:.2f} seconds")
        
        start_barrier.wait()
        size = (dataset[0].numel() * dataset[0].element_size() * len(dataset) / 1e9) # in GB

        for n in range(args.iters):
            # Start transfer
            start = time.perf_counter()
            start_transfer(
                agent_name,
                agent,
                transfer_handle,
                peer_name,
            )
            end = time.perf_counter()
            transfer_time = end - start
            transfer_speed = size / transfer_time
            logger.info(f"Round {n}: Transfer speed: {transfer_speed:.2f} GB/s")

        # Check the result
        if "peer" in args.role:
            for i, block in enumerate(dataset):
                assert torch.mean(block) - 1 < 1e-8, f"Block {i} not equal to 1"
            logger.info("Passed correctness check!")

        return transfer_speed
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C)")
        return 0.0  # or clean exit
    except Exception as e:
        logger.error(f"Error in agent pair {agent_name}: {traceback.format_exc()}")
        return 0.0
    finally:
        # Clean up transfer
        cleanup_transfer(
            agent,
            transfer_handle,
            register_descs,
        )
        # Clean up agent
        cleanup_agent(agent)

        # Close ZMQ socket
        zmq_socket.close()
        logger.info(f"Agent pair {agent_name} completed")

if __name__ == "__main__":
    args = parse_args()
    devices = []

    if args.device == "cpu":
        devices = ["cpu"]
    else:
        if torch.cuda.is_available():
            logger.info(f"CUDA is available with {torch.cuda.device_count()} device(s)")
            for i in range(torch.cuda.device_count()):
                devices.append(f'cuda:{i}')
        else:
            logger.warning("CUDA is not available, falling back to CPU")
            devices = ["cpu"]
    
    
    logger.info(f"Starting benchmark with {args.threads} threads")
    logger.info(f"Configuration: device={args.device}, num_blocks={args.num_blocks}, "
                f"num_layers={args.num_layers}, block_size={args.block_size}, "
                f"hidden_dim={args.hidden_dim}, dtype={args.dtype}")

    # Create a barrier to synchronize thread starts
    start_barrier = threading.Barrier(args.threads)
    
    # Create a shared variable to store results
    bw = []
    bw_lock = threading.Lock()

    def start_agents_test(agent_name, device, op, uid):
        result = start_agent_pair(agent_name, device, op, uid)
        with bw_lock:
            bw.append(result)

    # Create and start threads
    threads = [
        threading.Thread(
            target=start_agents_test,
            args=(f"{args.role}-{get_uuid()}-{i}", devices[i % len(devices)], args.operation, f"{get_uuid()}-{i}"),
            name=f"AgentPair-{i}"
        )
        for i in range(args.threads)
    ]
    
    for thread in threads:
        thread.start()
        logger.info(f"Started thread {thread.name}")
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
        logger.info(f"Thread {thread.name} completed")
    
    # Calculate and log the total transfer speed
    total_speed = sum(bw)
    avg_speed = total_speed / len(bw) if bw else 0.0
    logger.info(f"All threads completed successfully")
    logger.info(f"Total transfer speed across all threads: {total_speed:.2f} GB/s")
    logger.info(f"Average transfer speed per thread: {avg_speed:.2f} GB/s")

