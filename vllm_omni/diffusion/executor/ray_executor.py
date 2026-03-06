"""Ray-based distributed executor for diffusion models.

Enables multi-node GPU execution by using Ray actors instead of local multiprocessing.
Based on vLLM's RayDistributedExecutor pattern but adapted for diffusion models.

Key differences from multiprocessing executor:
- Workers are Ray actors that can span multiple nodes
- Distributed initialization uses TCP endpoint instead of localhost
- Communication happens via Ray actor method calls instead of MessageQueue
"""

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, get_open_port

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)

# Lazy import ray to avoid hard dependency
ray = None


def _ensure_ray():
    global ray
    if ray is None:
        import ray as _ray

        ray = _ray


@dataclass
class RayWorkerMetaData:
    """Metadata for tracking Ray worker actors."""

    worker: Any  # ActorHandle
    rank: int
    local_rank: int = 0
    ip: str = ""
    node_id: str = ""


class RayDiffusionWorkerWrapper:
    """Ray actor wrapper for diffusion workers.

    This class runs as a Ray actor. It lazily initializes the actual DiffusionWorker
    after Ray has set up the environment (CUDA_VISIBLE_DEVICES, etc.).

    Key responsibilities:
    - Initialize distributed environment with proper MASTER_ADDR/PORT
    - Create and manage the underlying DiffusionWorker
    - Handle method execution for RPC calls
    """

    def __init__(self, rpc_rank: int):
        """Initialize the wrapper with rpc_rank.

        Args:
            rpc_rank: The rank assigned during actor creation (may be reassigned later)
        """
        self.rpc_rank = rpc_rank
        self.rank = rpc_rank  # Will be updated in init_worker
        self.local_rank = 0  # Will be updated in init_worker
        self.worker = None
        self.od_config = None
        self._initialized = False

    def get_node_ip(self) -> str:
        """Get IP address of the node this actor is running on."""
        return get_ip()

    def get_node_and_gpu_ids(self) -> tuple[str, list[int]]:
        """Get node ID and GPU IDs assigned to this actor."""
        import ray as _ray  # Import directly in actor method

        node_id = _ray.get_runtime_context().get_node_id()
        # Get GPU IDs from Ray runtime context
        accelerator_ids = _ray.get_runtime_context().get_accelerator_ids()
        gpu_ids = accelerator_ids.get("GPU", [])
        return node_id, gpu_ids

    def update_environment_variables(self, env_vars: dict[str, str]) -> None:
        """Update environment variables for this actor."""
        os.environ.update(env_vars)
        logger.debug("Rank %d: Updated env vars: %s", self.rpc_rank, list(env_vars.keys()))

    def init_worker(
        self,
        od_config: OmniDiffusionConfig,
        rank: int,
        local_rank: int,
        distributed_init_method: str,
        worker_extension_cls: str | None = None,
        custom_pipeline_args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Initialize the actual diffusion worker.

        This is called after:
        1. Ray actor is created
        2. Environment variables (CUDA_VISIBLE_DEVICES) are set
        3. Worker distribution is determined

        Args:
            od_config: Diffusion configuration
            rank: Global rank (may differ from rpc_rank after reordering)
            local_rank: Local rank on the node
            distributed_init_method: TCP endpoint for torch.distributed init
            worker_extension_cls: Optional worker extension class
            custom_pipeline_args: Optional custom pipeline arguments
        """
        from vllm_omni.plugins import load_omni_general_plugins

        load_omni_general_plugins()

        self.rank = rank
        self.local_rank = local_rank
        self.od_config = od_config

        # Override the MASTER_ADDR/PORT with the distributed init method
        # The distributed_init_method is like "tcp://192.168.1.1:29500"
        # We need to set MASTER_ADDR and MASTER_PORT for torch.distributed
        if distributed_init_method.startswith("tcp://"):
            addr_port = distributed_init_method[6:]  # Remove "tcp://"
            if ":" in addr_port:
                master_addr, master_port = addr_port.rsplit(":", 1)
                os.environ["MASTER_ADDR"] = master_addr
                os.environ["MASTER_PORT"] = master_port
                logger.info(
                    "Rank %d: Setting MASTER_ADDR=%s, MASTER_PORT=%s",
                    rank,
                    master_addr,
                    master_port,
                )

        # Set rank-related environment variables
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(od_config.num_gpus)

        # Log GPU assignment for debugging
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET")
        logger.info(
            "Rank %d (local_rank %d): CUDA_VISIBLE_DEVICES = %s",
            rank,
            local_rank,
            cuda_devices,
        )

        # Create the worker using DiffusionWorker
        from vllm_omni.diffusion.worker import DiffusionWorker

        try:
            self.worker = DiffusionWorker(
                local_rank=local_rank,
                rank=rank,
                od_config=od_config,
            )
            self._initialized = True
            logger.info("Rank %d: Worker initialized successfully", rank)
            return {"status": "ready", "rank": rank}
        except Exception as e:
            logger.error("Rank %d: Worker initialization failed: %s", rank, e)
            return {"status": "error", "rank": rank, "error": str(e)}

    def execute_model(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        """Execute model inference.

        In diffusion with sequence parallelism (Ulysses/Ring), all workers
        execute the model simultaneously, but only rank 0 returns the result.
        """
        if not self._initialized or self.worker is None:
            raise RuntimeError("Worker not initialized. Call init_worker first.")

        # The worker.execute_model handles the distributed coordination internally
        result = self.worker.execute_model(request, self.od_config)

        # Move output tensors to CPU before returning to avoid OOM during Ray deserialization
        if result.output is not None:
            result.output = result.output.cpu()
        if result.trajectory_latents is not None:
            result.trajectory_latents = result.trajectory_latents.cpu()
        if result.trajectory_decoded is not None:
            result.trajectory_decoded = [t.cpu() for t in result.trajectory_decoded]
        if result.trajectory_timesteps is not None:
            result.trajectory_timesteps = [t.cpu() for t in result.trajectory_timesteps]

        return result

    def generate(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        """Alias for execute_model to match worker interface."""
        return self.execute_model(request)

    def execute_method(self, method: str, *args, **kwargs) -> Any:
        """Execute arbitrary method on the worker."""
        if not self._initialized or self.worker is None:
            raise RuntimeError("Worker not initialized. Call init_worker first.")
        return self.worker.execute_method(method, *args, **kwargs)

    def check_alive(self) -> bool:
        """Health check."""
        return self._initialized

    def shutdown(self) -> None:
        """Shutdown the worker."""
        if self.worker is not None:
            try:
                self.worker.shutdown()
            except Exception as e:
                logger.warning("Rank %d: Error during shutdown: %s", self.rank, e)
        self._initialized = False


class RayDiffusionExecutor(DiffusionExecutor):
    """Ray-based executor for distributed diffusion inference.

    This executor supports:
    - Single-node multi-GPU execution
    - Multi-node execution across a Ray cluster
    - Sequence parallelism (Ulysses/Ring) where all GPUs participate

    Key design decisions:
    1. All workers execute model simultaneously (unlike LLM where only some execute)
    2. Workers are sorted by IP to ensure workers on same node have consecutive ranks
    3. MASTER_ADDR is set to driver node's IP for torch.distributed init
    4. add_req broadcasts to all workers, collects result from rank 0
    """

    uses_ray: bool = True

    def _init_executor(self) -> None:
        _ensure_ray()

        self._closed = False
        self.workers: list[RayWorkerMetaData] = []

        # Initialize Ray if not already running
        if not ray.is_initialized():
            ray_address = getattr(self.od_config, "ray_address", None)
            if ray_address:
                logger.info("Connecting to Ray cluster at %s", ray_address)
                ray.init(address=ray_address)
            else:
                logger.info("Initializing local Ray instance")
                ray.init()

        # Create placement group for GPU resources
        placement_group = self._create_placement_group()

        # Launch and initialize worker actors
        self._init_workers_ray(placement_group)

    def _create_placement_group(self):
        """Create placement group for GPU allocation."""
        num_gpus = self.od_config.num_gpus

        # Check if we're already in a placement group
        current_pg = ray.util.get_current_placement_group()
        if current_pg is not None:
            logger.info("Using existing placement group")
            return current_pg

        # Create a new placement group
        # Each bundle requests 1 GPU and 1 CPU (workers need both)
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]

        # Try to get the current node's IP to prefer local placement
        driver_ip = get_ip()
        if bundles:
            # Request at least one bundle on the driver node
            bundles[0][f"node:{driver_ip}"] = 0.001

        placement_group = ray.util.placement_group(bundles, strategy="PACK")

        # Wait for placement group to be ready
        logger.info("Waiting for placement group with %d GPU bundles...", num_gpus)
        ray.get(placement_group.ready(), timeout=300)
        logger.info("Placement group ready")

        return placement_group

    def _init_workers_ray(self, placement_group) -> None:
        """Initialize Ray worker actors with proper distributed setup."""
        num_gpus = self.od_config.num_gpus
        driver_ip = get_ip()

        # Create Ray remote class
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        ray_remote_kwargs = {
            "num_gpus": 1,
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
            ),
        }

        RemoteWorker = ray.remote(**ray_remote_kwargs)(RayDiffusionWorkerWrapper)

        # Spawn worker actors
        worker_metadata: list[RayWorkerMetaData] = []
        for rank in range(num_gpus):
            actor = RemoteWorker.remote(rpc_rank=rank)
            worker_metadata.append(RayWorkerMetaData(worker=actor, rank=rank))

        # Get worker IPs and node info
        # Collect all ObjectRefs in flat lists (ray.get doesn't accept tuples)
        ip_refs = [w.worker.get_node_ip.remote() for w in worker_metadata]
        node_gpu_refs = [w.worker.get_node_and_gpu_ids.remote() for w in worker_metadata]

        ips = ray.get(ip_refs)
        node_gpu_infos = ray.get(node_gpu_refs)

        for meta, ip, (node_id, gpu_ids) in zip(worker_metadata, ips, node_gpu_infos):
            meta.ip = ip
            meta.node_id = node_id

        # Sort workers: driver node first, then by IP
        ip_counts: dict[str, int] = defaultdict(int)
        for meta in worker_metadata:
            ip_counts[meta.ip] += 1

        def sort_key(meta: RayWorkerMetaData):
            # Driver node first, then by number of workers on node, then by IP
            return (0 if meta.ip == driver_ip else 1, ip_counts[meta.ip], meta.ip)

        sorted_metadata = sorted(worker_metadata, key=sort_key)

        # In Ray, each actor gets exactly 1 GPU via num_gpus=1 in ray.remote()
        # Ray sets CUDA_VISIBLE_DEVICES automatically to only show that GPU
        # So local_rank is always 0 (each actor sees only its assigned GPU)
        for i, meta in enumerate(sorted_metadata):
            meta.rank = i
            meta.local_rank = 0  # Always 0 since Ray isolates GPUs per actor

        self.workers = sorted_metadata

        logger.info("Worker distribution:")
        for meta in self.workers:
            logger.info(
                "  rank=%d, local_rank=%d, ip=%s, node_id=%s",
                meta.rank,
                meta.local_rank,
                meta.ip,
                meta.node_id[:8] if meta.node_id else "N/A",
            )

        # Note: We do NOT override CUDA_VISIBLE_DEVICES - Ray handles GPU assignment
        # Each actor sees only its assigned GPU at index 0

        # Determine distributed init method
        # Use 127.0.0.1 for single-node, driver IP for multi-node
        unique_nodes = len(set(meta.node_id for meta in self.workers))
        if unique_nodes == 1:
            master_addr = "127.0.0.1"
        else:
            master_addr = driver_ip

        master_port = get_open_port()
        distributed_init_method = f"tcp://{master_addr}:{master_port}"
        logger.info("Distributed init method: %s", distributed_init_method)

        # Initialize all workers
        worker_extension_cls = self.od_config.worker_extension_cls
        custom_pipeline_args = getattr(self.od_config, "custom_pipeline_args", None)

        init_futures = []
        for meta in self.workers:
            future = meta.worker.init_worker.remote(
                od_config=self.od_config,
                rank=meta.rank,
                local_rank=meta.local_rank,
                distributed_init_method=distributed_init_method,
                worker_extension_cls=worker_extension_cls,
                custom_pipeline_args=custom_pipeline_args,
            )
            init_futures.append(future)

        # Wait for all workers to initialize
        try:
            results = ray.get(init_futures, timeout=600)  # 10 min timeout for model loading
            for result in results:
                if result.get("status") != "ready":
                    raise RuntimeError(f"Worker initialization failed: {result}")
            logger.info("All %d workers initialized successfully", len(self.workers))
        except Exception as e:
            logger.error("Worker initialization failed: %s", e)
            self.shutdown()
            raise

    def add_req(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        """Add a diffusion request and get the output.

        For diffusion with sequence parallelism, ALL workers must execute
        the model simultaneously. This method broadcasts the request to all
        workers and collects the result from rank 0.
        """
        if self._closed:
            raise RuntimeError("RayDiffusionExecutor is closed.")

        # Broadcast request to ALL workers (required for sequence parallelism)
        # All workers participate in the distributed computation
        futures = [meta.worker.execute_model.remote(request) for meta in self.workers]

        try:
            # Wait for all workers to complete
            results = ray.get(futures, timeout=600)
            # Return result from rank 0
            return results[0]
        except ray.exceptions.RayTaskError as e:
            logger.error("Worker execution failed: %s", e)
            raise RuntimeError(f"Diffusion generation failed: {e}") from e
        except ray.exceptions.GetTimeoutError as e:
            logger.error("Worker execution timed out")
            raise TimeoutError("Diffusion generation timed out") from e

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Execute RPC on workers.

        Args:
            method: Method name to call on workers.
            timeout: Timeout in seconds.
            args: Positional arguments for the method.
            kwargs: Keyword arguments for the method.
            unique_reply_rank: If set, only get response from this rank.

        Returns:
            Single response if unique_reply_rank is set, else list of responses.
        """
        if self._closed:
            raise RuntimeError("RayDiffusionExecutor is closed.")

        kwargs = kwargs or {}
        timeout = timeout or 300  # Default 5 min timeout

        # Always call all workers (required for collective operations)
        futures = [
            meta.worker.execute_method.remote(method, *args, **kwargs)
            for meta in self.workers
        ]

        # Collect responses
        try:
            responses = ray.get(futures, timeout=timeout)
            if unique_reply_rank is not None:
                return responses[unique_reply_rank]
            return responses
        except ray.exceptions.GetTimeoutError as e:
            raise TimeoutError(f"RPC call to {method} timed out.") from e
        except ray.exceptions.RayTaskError as e:
            logger.error("RPC call failed: %s", e)
            raise RuntimeError(f"RPC call to {method} failed: {e}") from e

    def check_health(self) -> None:
        """Check if all worker actors are alive."""
        if self._closed:
            raise RuntimeError("RayDiffusionExecutor is closed.")

        for meta in self.workers:
            try:
                alive = ray.get(meta.worker.check_alive.remote(), timeout=10)
                if not alive:
                    raise RuntimeError(f"Worker rank {meta.rank} is not healthy")
            except Exception as e:
                raise RuntimeError(f"Worker rank {meta.rank} health check failed: {e}") from e

    def shutdown(self) -> None:
        """Shutdown all workers and cleanup."""
        if self._closed:
            return

        self._closed = True
        logger.info("Shutting down RayDiffusionExecutor...")

        # First try graceful shutdown
        shutdown_futures = []
        for meta in self.workers:
            try:
                shutdown_futures.append(meta.worker.shutdown.remote())
            except Exception as e:
                logger.warning("Failed to send shutdown to worker rank %d: %s", meta.rank, e)

        # Wait briefly for graceful shutdown
        if shutdown_futures:
            try:
                ray.get(shutdown_futures, timeout=10)
            except Exception:
                pass  # Ignore errors during shutdown

        # Force kill actors
        for meta in self.workers:
            try:
                ray.kill(meta.worker)
            except Exception as e:
                logger.warning("Failed to kill worker rank %d: %s", meta.rank, e)

        self.workers.clear()
        logger.info("RayDiffusionExecutor shutdown complete")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.shutdown()
        except Exception:
            pass
