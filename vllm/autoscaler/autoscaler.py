import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union

from kubernetes import client, config
from kubernetes.client.rest import ApiException

from vllm.redis.redis_client import RedisClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vllm.autoscaler")


class ScalingPolicy:
    """Base class for scaling policies."""

    def __init__(self, name: str):
        """Initialize the scaling policy.

        Args:
            name: Name of the policy.
        """
        self.name = name

    def should_scale(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Determine if scaling is needed based on metrics.

        Args:
            metrics: Dictionary of metrics.

        Returns:
            Dictionary with scaling decision and details.
        """
        raise NotImplementedError("Subclasses must implement should_scale")


class QueueLengthPolicy(ScalingPolicy):
    """Scaling policy based on queue length."""

    def __init__(
        self,
        min_replicas: int = 1,
        max_replicas: int = 10,
        target_queue_length: int = 5,
        scale_up_threshold: int = 10,
        scale_down_threshold: int = 2,
        cooldown_period: int = 60,
    ):
        """Initialize the queue length policy.

        Args:
            min_replicas: Minimum number of replicas.
            max_replicas: Maximum number of replicas.
            target_queue_length: Target queue length per replica.
            scale_up_threshold: Queue length threshold for scaling up.
            scale_down_threshold: Queue length threshold for scaling down.
            cooldown_period: Cooldown period in seconds between scaling actions.
        """
        super().__init__("queue_length")
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.target_queue_length = target_queue_length
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        self.last_scale_time = 0

    def should_scale(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Determine if scaling is needed based on queue length.

        Args:
            metrics: Dictionary of metrics including queue_length and
                current_replicas.

        Returns:
            Dictionary with scaling decision and details.
        """
        current_time = time.time()
        if current_time - self.last_scale_time < self.cooldown_period:
            return {
                "scale": False,
                "reason": "In cooldown period",
                "policy": self.name,
            }

        queue_length = metrics.get("queue_length", 0)
        current_replicas = metrics.get("current_replicas", 0)

        if current_replicas == 0:
            # No replicas running, start with min_replicas
            return {
                "scale": True,
                "target_replicas": self.min_replicas,
                "reason": f"No replicas running, starting with {self.min_replicas}",
                "policy": self.name,
            }

        queue_length_per_replica = queue_length / current_replicas

        if queue_length_per_replica > self.scale_up_threshold:
            # Queue is growing too fast, need to scale up
            target_replicas = min(
                self.max_replicas,
                max(
                    self.min_replicas,
                    int(queue_length / self.target_queue_length)
                )
            )

            if target_replicas > current_replicas:
                self.last_scale_time = current_time
                return {
                    "scale": True,
                    "target_replicas": target_replicas,
                    "reason": (
                        f"Queue length per replica "
                        f"({queue_length_per_replica:.2f}) "
                        f"exceeds scale-up threshold "
                        f"({self.scale_up_threshold})"
                    ),
                    "policy": self.name,
                }

        elif queue_length_per_replica < self.scale_down_threshold:
            # Queue is small, can scale down
            target_replicas = max(
                self.min_replicas,
                min(
                    current_replicas - 1,
                    max(1, int(queue_length / self.target_queue_length))
                )
            )

            if target_replicas < current_replicas:
                self.last_scale_time = current_time
                return {
                    "scale": True,
                    "target_replicas": target_replicas,
                    "reason": (
                        f"Queue length per replica ({queue_length_per_replica:.2f}) "
                        f"below scale-down threshold "
                        f"({self.scale_down_threshold})"
                    ),
                    "policy": self.name,
                }

        return {
            "scale": False,
            "reason": (
                f"Queue length per replica ({queue_length_per_replica:.2f}) "
                f"within thresholds ({self.scale_down_threshold}-"
                f"{self.scale_up_threshold})"
            ),
            "policy": self.name,
        }


class LatencyPolicy(ScalingPolicy):
    """Scaling policy based on request latency."""

    def __init__(
        self,
        min_replicas: int = 1,
        max_replicas: int = 10,
        target_latency: float = 1.0,
        scale_up_threshold: float = 2.0,
        scale_down_threshold: float = 0.5,
        cooldown_period: int = 60,
    ):
        """Initialize the latency policy.

        Args:
            min_replicas: Minimum number of replicas.
            max_replicas: Maximum number of replicas.
            target_latency: Target latency in seconds.
            scale_up_threshold: Latency threshold for scaling up.
            scale_down_threshold: Latency threshold for scaling down.
            cooldown_period: Cooldown period in seconds between scaling actions.
        """
        super().__init__("latency")
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.target_latency = target_latency
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        self.last_scale_time = 0

    def should_scale(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Determine if scaling is needed based on latency.

        Args:
            metrics: Dictionary of metrics including latency and current_replicas.

        Returns:
            Dictionary with scaling decision and details.
        """
        current_time = time.time()
        if current_time - self.last_scale_time < self.cooldown_period:
            return {
                "scale": False,
                "reason": "In cooldown period",
                "policy": self.name,
            }

        latency = metrics.get("latency", 0)
        current_replicas = metrics.get("current_replicas", 0)

        if current_replicas == 0:
            # No replicas running, start with min_replicas
            return {
                "scale": True,
                "target_replicas": self.min_replicas,
                "reason": f"No replicas running, starting with {self.min_replicas}",
                "policy": self.name,
            }

        if latency > self.target_latency * self.scale_up_threshold:
            # Latency is too high, need to scale up
            target_replicas = min(
                self.max_replicas,
                current_replicas + 1
            )

            if target_replicas > current_replicas:
                self.last_scale_time = current_time
                return {
                    "scale": True,
                    "target_replicas": target_replicas,
                    "reason": (
                        f"Latency ({latency:.2f}s) exceeds "
                        f"scale-up threshold "
                        f"({self.target_latency * self.scale_up_threshold:.2f}s)"
                    ),
                    "policy": self.name,
                }

        elif latency < self.target_latency * self.scale_down_threshold:
            # Latency is low, can scale down
            target_replicas = max(
                self.min_replicas,
                current_replicas - 1
            )

            if target_replicas < current_replicas:
                self.last_scale_time = current_time
                return {
                    "scale": True,
                    "target_replicas": target_replicas,
                    "reason": (
                        f"Latency ({latency:.2f}s) below scale-down threshold "
                        f"({self.target_latency * self.scale_down_threshold:.2f}s)"
                    ),
                    "policy": self.name,
                }

        return {
            "scale": False,
            "reason": (
                f"Latency ({latency:.2f}s) within thresholds "
                f"({self.target_latency * self.scale_down_threshold:.2f}s-"
                f"{self.target_latency * self.scale_up_threshold:.2f}s)"
            ),
            "policy": self.name,
        }


class KubernetesScaler:
    """Kubernetes implementation of the scaler."""

    def __init__(
        self,
        namespace: str,
        deployment_name: str,
        min_replicas: int = 1,
        max_replicas: int = 10,
    ):
        """Initialize the Kubernetes scaler.

        Args:
            namespace: Kubernetes namespace.
            deployment_name: Name of the deployment to scale.
            min_replicas: Minimum number of replicas.
            max_replicas: Maximum number of replicas.
        """
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas

        try:
            # Load kubeconfig
            config.load_kube_config()
            self.apps_v1 = client.AppsV1Api()
            logger.info("Initialized Kubernetes client")
        except Exception as e:
            logger.error(
                f"Failed to initialize Kubernetes client: "
                f"{e}"
            )
            raise

    def get_current_replicas(self) -> int:
        """Get the current number of replicas.

        Returns:
            Current number of replicas.
        """
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace
            )
            return deployment.spec.replicas
        except ApiException as e:
            logger.error(
                f"Failed to get deployment: "
                f"{e}"
            )
            return 0

    def scale(self, replicas: int) -> bool:
        """Scale the deployment to the specified number of replicas.

        Args:
            replicas: Target number of replicas.

        Returns:
            True if scaling was successful, False otherwise.
        """
        # Ensure replicas is within bounds
        replicas = max(self.min_replicas, min(self.max_replicas, replicas))

        try:
            # Update the deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace
            )

            if deployment.spec.replicas == replicas:
                logger.info(
                    f"Deployment {self.deployment_name} already has "
                    f"{replicas} replicas"
                )
                return True

            # Update the replicas
            deployment.spec.replicas = replicas
            self.apps_v1.patch_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace,
                body=deployment
            )

            logger.info(
                f"Scaled deployment {self.deployment_name} "
                f"to {replicas} replicas"
            )
            return True

        except ApiException as e:
            logger.error(
                f"Failed to scale deployment: "
                f"{e}"
            )
            return False


class SkyPilotScaler:
    """SkyPilot implementation of the scaler."""

    def __init__(
        self,
        cluster_name: str,
        min_replicas: int = 1,
        max_replicas: int = 10,
        skypilot_cmd: str = "sky",
    ):
        """Initialize the SkyPilot scaler.

        Args:
            cluster_name: Name of the SkyPilot cluster.
            min_replicas: Minimum number of replicas.
            max_replicas: Maximum number of replicas.
            skypilot_cmd: Command to run SkyPilot.
        """
        self.cluster_name = cluster_name
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.skypilot_cmd = skypilot_cmd

        # Check if SkyPilot is installed
        try:
            import subprocess
            result = subprocess.run(
                [self.skypilot_cmd, "status"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.warning(
                    f"SkyPilot command failed: {result.stderr}. "
                    "Scaling operations may not work."
                )
            else:
                logger.info("Initialized SkyPilot scaler")
        except Exception as e:
            logger.error(f"Failed to initialize SkyPilot scaler: {e}")
            raise

    def get_current_replicas(self) -> int:
        """Get the current number of replicas.

        Returns:
            Current number of replicas.
        """
        try:
            import subprocess
            result = subprocess.run(
                [self.skypilot_cmd, "status", self.cluster_name],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.error(
                    f"Failed to get cluster status: "
                    f"{result.stderr}"
                )
                return 0

            # Parse the output to get the number of replicas
            # This is a simplified implementation and may need to be adjusted
            # based on the actual output format of SkyPilot
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if "WORKERS" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "WORKERS":
                            return int(parts[i+1])

            return 0

        except Exception as e:
            logger.error(f"Failed to get current replicas: {e}")
            return 0

    def scale(self, replicas: int) -> bool:
        """Scale the SkyPilot cluster to the specified number of replicas.

        Args:
            replicas: Target number of replicas.

        Returns:
            True if scaling was successful, False otherwise.
        """
        # Ensure replicas is within bounds
        replicas = max(self.min_replicas, min(self.max_replicas, replicas))

        try:
            import subprocess

            current_replicas = self.get_current_replicas()
            if current_replicas == replicas:
                logger.info(
                    f"Cluster {self.cluster_name} already has {replicas} replicas"
                )
                return True

            # Scale the cluster
            result = subprocess.run(
                [
                    self.skypilot_cmd, "scale", self.cluster_name,
                    "--workers", str(replicas)
                ],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.error(
                    f"Failed to scale cluster: "
                    f"{result.stderr}"
                )
                return False

            logger.info(
                f"Scaled cluster {self.cluster_name} "
                f"to {replicas} replicas"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to scale cluster: "
                f"{e}"
            )
            return False


class VLLMAutoscaler:
    """Autoscaler for vLLM models."""

    def __init__(
        self,
        model_name: str,
        redis_client: RedisClient,
        scaler: Union[KubernetesScaler, SkyPilotScaler],
        policies: Optional[List[ScalingPolicy]] = None,
        metrics_collection_interval: int = 10,
        scaling_interval: int = 30,
    ):
        """Initialize the vLLM autoscaler.

        Args:
            model_name: Name of the model to autoscale.
            redis_client: RedisClient instance for Redis communication.
            scaler: Scaler implementation (Kubernetes or SkyPilot).
            policies: List of scaling policies. If None, default policies will be used.
            metrics_collection_interval: Interval in seconds to collect metrics.
            scaling_interval: Interval in seconds to make scaling decisions.
        """
        self.model_name = model_name
        self.redis_client = redis_client
        self.scaler = scaler

        # Use default policies if none provided
        if policies is None:
            self.policies = [
                QueueLengthPolicy(),
                LatencyPolicy(),
            ]
        else:
            self.policies = policies

        self.metrics_collection_interval = metrics_collection_interval
        self.scaling_interval = scaling_interval
        self.running = False
        self.metrics_task = None
        self.scaling_task = None
        self.metrics = {}

    async def start(self) -> None:
        """Start the autoscaler."""
        if self.running:
            return

        self.running = True
        self.metrics_task = asyncio.create_task(self._collect_metrics())
        self.scaling_task = asyncio.create_task(self._run_scaling_loop())
        logger.info(f"Started autoscaler for model {self.model_name}")

    async def stop(self) -> None:
        """Stop the autoscaler."""
        if not self.running:
            return

        self.running = False

        if self.metrics_task:
            self.metrics_task.cancel()
            try:
                await self.metrics_task
            except asyncio.CancelledError:
                pass
            self.metrics_task = None

        if self.scaling_task:
            self.scaling_task.cancel()
            try:
                await self.scaling_task
            except asyncio.CancelledError:
                pass
            self.scaling_task = None

        logger.info(f"Stopped autoscaler for model {self.model_name}")

    async def _collect_metrics(self) -> None:
        """Collect metrics from Redis."""
        while self.running:
            try:
                # Get metrics from Redis
                redis_metrics = self.redis_client.get_metrics(self.model_name)

                # Get queue length
                queue_length = self.redis_client.get_queue_length(
                    self.model_name)

                # Get current number of replicas
                current_replicas = self.scaler.get_current_replicas()

                # Update metrics
                self.metrics.update({
                    "queue_length": queue_length,
                    "current_replicas": current_replicas,
                    "last_updated": time.time(),
                    **redis_metrics,
                })

                logger.debug(f"Collected metrics: {self.metrics}")

            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")

            await asyncio.sleep(self.metrics_collection_interval)

    async def _run_scaling_loop(self) -> None:
        """Run the scaling loop."""
        while self.running:
            try:
                if not self.metrics:
                    logger.warning("No metrics available for scaling decision")
                    await asyncio.sleep(self.scaling_interval)
                    continue

                # Apply each policy and get scaling decisions
                decisions = [
                    policy.should_scale(self.metrics) for policy in self.policies
                ]

                # Filter decisions that recommend scaling
                scale_decisions = [
                    d for d in decisions if d.get("scale", False)]

                if scale_decisions:
                    # Choose the decision with the highest priority
                    # For now, we'll just take the first one
                    decision = scale_decisions[0]

                    # Scale the deployment
                    target_replicas = decision.get("target_replicas", 1)
                    success = self.scaler.scale(target_replicas)

                    if success:
                        logger.info(
                            f"Scaled to {target_replicas} replicas. "
                            f"Reason: {decision.get('reason')}"
                        )
                    else:
                        logger.error(
                            f"Failed to scale to {target_replicas} replicas"
                        )
                else:
                    logger.debug("No scaling needed")

            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")

            await asyncio.sleep(self.scaling_interval)

    def get_metrics(self) -> Dict[str, Any]:
        """Get the current metrics.

        Returns:
            Dictionary of metrics.
        """
        return self.metrics.copy()


async def run_autoscaler(
    model_name: str,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    redis_password: Optional[str] = None,
    scaler_type: str = "kubernetes",
    namespace: str = "default",
    deployment_name: str = "vllm",
    cluster_name: str = "vllm-cluster",
    min_replicas: int = 1,
    max_replicas: int = 10,
    metrics_collection_interval: int = 10,
    scaling_interval: int = 30,
) -> None:
    """Run the vLLM autoscaler.

    Args:
        model_name: Name of the model to autoscale.
        redis_host: Redis host.
        redis_port: Redis port.
        redis_db: Redis database.
        redis_password: Redis password.
        scaler_type: Type of scaler to use ("kubernetes" or "skypilot").
        namespace: Kubernetes namespace.
        deployment_name: Name of the Kubernetes deployment.
        cluster_name: Name of the SkyPilot cluster.
        min_replicas: Minimum number of replicas.
        max_replicas: Maximum number of replicas.
        metrics_collection_interval: Interval in seconds to collect metrics.
        scaling_interval: Interval in seconds to make scaling decisions.
    """
    # Initialize Redis client
    redis_client = RedisClient(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        password=redis_password,
    )

    # Initialize scaler
    if scaler_type.lower() == "kubernetes":
        scaler = KubernetesScaler(
            namespace=namespace,
            deployment_name=deployment_name,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
        )
    elif scaler_type.lower() == "skypilot":
        scaler = SkyPilotScaler(
            cluster_name=cluster_name,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
        )
    else:
        raise ValueError(
            f"Invalid scaler type: {scaler_type}. "
            "Must be 'kubernetes' or 'skypilot'"
        )

    # Initialize autoscaler
    autoscaler = VLLMAutoscaler(
        model_name=model_name,
        redis_client=redis_client,
        scaler=scaler,
        metrics_collection_interval=metrics_collection_interval,
        scaling_interval=scaling_interval,
    )

    # Start the autoscaler
    await autoscaler.start()

    try:
        # Keep the autoscaler running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping autoscaler...")
    finally:
        # Stop the autoscaler
        await autoscaler.stop()


def main():
    """Main entry point for the autoscaler."""
    import argparse

    parser = argparse.ArgumentParser(description="vLLM Autoscaler")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to autoscale"
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default="localhost",
        help="Redis host"
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port"
    )
    parser.add_argument(
        "--redis-db",
        type=int,
        default=0,
        help="Redis database"
    )
    parser.add_argument(
        "--redis-password",
        type=str,
        default=None,
        help="Redis password"
    )
    parser.add_argument(
        "--scaler-type",
        type=str,
        choices=["kubernetes", "skypilot"],
        default="kubernetes",
        help="Type of scaler to use"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Kubernetes namespace"
    )
    parser.add_argument(
        "--deployment-name",
        type=str,
        default="vllm",
        help="Name of the Kubernetes deployment"
    )
    parser.add_argument(
        "--cluster-name",
        type=str,
        default="vllm-cluster",
        help="Name of the SkyPilot cluster"
    )
    parser.add_argument(
        "--min-replicas",
        type=int,
        default=1,
        help="Minimum number of replicas"
    )
    parser.add_argument(
        "--max-replicas",
        type=int,
        default=10,
        help="Maximum number of replicas"
    )
    parser.add_argument(
        "--metrics-interval",
        type=int,
        default=10,
        help="Interval in seconds to collect metrics"
    )
    parser.add_argument(
        "--scaling-interval",
        type=int,
        default=30,
        help="Interval in seconds to make scaling decisions"
    )

    args = parser.parse_args()

    # Run the autoscaler
    asyncio.run(run_autoscaler(
        model_name=args.model_name,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        redis_password=args.redis_password,
        scaler_type=args.scaler_type,
        namespace=args.namespace,
        deployment_name=args.deployment_name,
        cluster_name=args.cluster_name,
        min_replicas=args.min_replicas,
        max_replicas=args.max_replicas,
        metrics_collection_interval=args.metrics_interval,
        scaling_interval=args.scaling_interval,
    ))


if __name__ == "__main__":
    main()
