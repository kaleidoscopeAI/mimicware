import ray
from typing import List, Dict, Any, Optional
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.rpc import RRef
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import asyncio
import zmq
import zmq.asyncio
import logging

# Add missing imports
from torch.distributed.optim import DistributedOptimizer
from torch.optim import Adam

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DistributedRunner")

@dataclass
class ClusterConfig:
    num_workers: int
    memory_per_worker: int
    input_dim: int
    hidden_dim: int
    batch_size: int

# Add missing class definition
class KaleidoscopeEngine(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim//2)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim//2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

@ray.remote
class DistributedNode:
    def __init__(self, rank: int, world_size: int, config: ClusterConfig):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self._initialize_distributed()
        self.processor = self._create_processor()
        
    def _initialize_distributed(self):
        try:
            # Initialize process group
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '29500'
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                rank=self.rank,
                world_size=self.world_size
            )
            logger.info(f"Node {self.rank} initialized process group")
        except Exception as e:
            logger.error(f"Error initializing distributed: {e}")
            raise
        
    def _create_processor(self) -> torch.nn.Module:
        device = torch.device(f'cuda:{self.rank % torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu')
        model = KaleidoscopeEngine(self.config.input_dim, self.config.hidden_dim).to(device)
        return DistributedDataParallel(model)
        
    async def process_data(self, data: torch.Tensor) -> torch.Tensor:
        try:
            device = next(self.processor.parameters()).device
            data = data.to(device)
            with torch.no_grad():
                result = self.processor(data)
            return result.cpu()
        except Exception as e:
            logger.error(f"Error processing data on node {self.rank}: {e}")
            return torch.zeros_like(data)
            
    def cleanup(self):
        try:
            dist.destroy_process_group()
            logger.info(f"Node {self.rank} cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

class DistributedCluster:
    def __init__(self, config: ClusterConfig):
        self.config = config
        ray.init(ignore_reinit_error=True)
        self.nodes: List[DistributedNode] = []
        self.initialize_cluster()
        
    def initialize_cluster(self):
        for i in range(self.config.num_workers):
            node = DistributedNode.remote(i, self.config.num_workers, self.config)
            self.nodes.append(node)
            logger.info(f"Initialized node {i}")
            
    async def process_batch(self, batch: torch.Tensor) -> List[torch.Tensor]:
        try:
            futures = [node.process_data.remote(batch) for node in self.nodes]
            results = await asyncio.gather(*[ray.get(future) for future in futures])
            return results
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return [torch.zeros_like(batch) for _ in range(len(self.nodes))]
        
    def shutdown(self):
        for node in self.nodes:
            ray.get(node.cleanup.remote())
        ray.shutdown()
        logger.info("Cluster shutdown completed")

# Add missing function
def create_kaleidoscope_ai():
    from kaleidoscope_ai import KaleidoscopeAI
    return KaleidoscopeAI(input_dim=512, hidden_dim=1024)

class KaleidoscopeRunner:
    def __init__(self, data_path: Path, config: ClusterConfig):
        self.data_path = data_path
        self.config = config
        self.cluster = DistributedCluster(config)
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")
        logger.info(f"Runner initialized with data path: {data_path}")
        
    def load_data(self) -> torch.utils.data.DataLoader:
        try:
            # Handle various file formats
            if self.data_path.suffix == '.pt' or self.data_path.suffix == '.pth':
                dataset = torch.load(self.data_path)
            elif self.data_path.suffix == '.npy':
                dataset = torch.tensor(np.load(self.data_path))
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
                
            # Wrap as Dataset if it's just a tensor
            if isinstance(dataset, torch.Tensor):
                dataset = torch.utils.data.TensorDataset(dataset)
                
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4
            )
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
        
    async def run_processing(self):
        try:
            data_loader = self.load_data()
            kaleidoscope = create_kaleidoscope_ai()
            
            for batch_idx, batch in enumerate(data_loader):
                # Ensure batch is a tensor
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]  # Take the first element if it's a tuple/list
                
                logger.info(f"Processing batch {batch_idx}")
                
                # Distribute batch processing
                processed_batches = await self.cluster.process_batch(batch)
                
                # Aggregate results
                combined_batch = torch.stack(processed_batches).mean(0)
                
                # Feed into KaleidoscopeAI
                results = kaleidoscope.process_data(combined_batch)
                logger.info(f"Processed batch {batch_idx} with supernode ID: {results['supernode'].id}")
                
                # Handle chat interface
                await self.handle_chat_interface(kaleidoscope)
                
            self.cluster.shutdown()
            logger.info("Processing completed")
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            self.cluster.shutdown()
            
    async def handle_chat_interface(self, kaleidoscope):
        while True:
            try:
                logger.info("Waiting for chat message...")
                message = await self.socket.recv_string()
                if message == "EXIT":
                    logger.info("Received EXIT command")
                    break
                
                # Use process_data if chat_interface doesn't exist
                if hasattr(kaleidoscope, 'chat_interface'):
                    response = kaleidoscope.chat_interface(message)
                else:
                    logger.warning("Using process_data as fallback for chat_interface")
                    dummy_tensor = torch.randn(1, 512)  # Create a dummy input tensor
                    results = kaleidoscope.process_data(dummy_tensor)
                    response = f"Processed with supernode ID: {results['supernode'].id}"
                
                await self.socket.send_string(response)
                
            except Exception as e:
                logger.error(f"Error in chat interface: {e}")
                await self.socket.send_string(f"Error: {str(e)}")
        
    @staticmethod
    async def create_and_run(data_path: str, num_workers: int = 4):
        import os
        # Ensure paths exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        config = ClusterConfig(
            num_workers=num_workers,
            memory_per_worker=8192,
            input_dim=512,
            hidden_dim=1024,
            batch_size=32
        )
        
        runner = KaleidoscopeRunner(Path(data_path), config)
        await runner.run_processing()

if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python distributed-runner.py <data_path> [num_workers]")
        sys.exit(1)
        
    data_path = sys.argv[1]
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    
    asyncio.run(KaleidoscopeRunner.create_and_run(data_path, num_workers))
