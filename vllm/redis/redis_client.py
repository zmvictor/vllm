import json
import time
from typing import Dict, List, Optional, Any, Tuple

import redis


class RedisClient:
    """Client for interacting with Redis Streams for vLLM request queuing."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "vllm:",
    ):
        """Initialize the Redis client.

        Args:
            host: Redis host.
            port: Redis port.
            db: Redis database.
            password: Redis password.
            prefix: Prefix for Redis keys.
        """
        self.redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
        )
        self.prefix = prefix

    def get_model_stream_key(self, model_name: str) -> str:
        """Get the Redis stream key for a model.

        Args:
            model_name: Name of the model.

        Returns:
            Redis stream key.
        """
        return f"{self.prefix}model:{model_name}:requests"

    def get_model_metrics_key(self, model_name: str) -> str:
        """Get the Redis key for model metrics.

        Args:
            model_name: Name of the model.

        Returns:
            Redis key for model metrics.
        """
        return f"{self.prefix}model:{model_name}:metrics"

    def add_request(
        self,
        model_name: str,
        request_id: str,
        prompt: Optional[str],
        sampling_params: Dict[str, Any],
        prompt_token_ids: Optional[List[int]] = None,
    ) -> str:
        """Add a request to the model's stream.

        Args:
            model_name: Name of the model.
            request_id: Unique ID for the request.
            prompt: The prompt string.
            sampling_params: Sampling parameters.
            prompt_token_ids: Token IDs for the prompt.

        Returns:
            ID of the message in the stream.
        """
        stream_key = self.get_model_stream_key(model_name)
        request_data = {
            "request_id": request_id,
            "prompt": prompt,
            "sampling_params": json.dumps(sampling_params),
            "prompt_token_ids": (
                json.dumps(prompt_token_ids) if prompt_token_ids else None
            ),
            "arrival_time": time.time(),
        }
        return self.redis.xadd(stream_key, request_data)

    def get_requests(
        self,
        model_name: str,
        count: int = 10,
        block: int = 0,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Get requests from the model's stream.

        Args:
            model_name: Name of the model.
            count: Maximum number of requests to get.
            block: Time to block in milliseconds. 0 means no blocking.

        Returns:
            List of (message_id, request_data) tuples.
        """
        stream_key = self.get_model_stream_key(model_name)
        # Get the latest message ID
        latest_id = "0-0"
        messages = self.redis.xread(
            {stream_key: latest_id}, count=count, block=block
        )
        if not messages:
            return []

        result = []
        for stream_name, stream_messages in messages:
            for message_id, data in stream_messages:
                # Convert sampling_params and prompt_token_ids back to Python
                # objects
                if data.get("sampling_params"):
                    data["sampling_params"] = json.loads(
                        data["sampling_params"])
                if data.get("prompt_token_ids"):
                    data["prompt_token_ids"] = json.loads(
                        data["prompt_token_ids"])
                result.append((message_id, data))
        return result

    def acknowledge_request(self, model_name: str, message_id: str) -> int:
        """Acknowledge that a request has been processed.

        Args:
            model_name: Name of the model.
            message_id: ID of the message in the stream.

        Returns:
            Number of messages acknowledged.
        """
        stream_key = self.get_model_stream_key(model_name)
        return self.redis.xdel(stream_key, message_id)

    def update_metrics(self, model_name: str, metrics: Dict[str, Any]) -> None:
        """Update metrics for a model.

        Args:
            model_name: Name of the model.
            metrics: Metrics to update.
        """
        metrics_key = self.get_model_metrics_key(model_name)
        # Convert all values to strings for Redis
        string_metrics = {k: str(v) for k, v in metrics.items()}
        self.redis.hset(metrics_key, mapping=string_metrics)

    def get_metrics(self, model_name: str) -> Dict[str, Any]:
        """Get metrics for a model.

        Args:
            model_name: Name of the model.

        Returns:
            Model metrics.
        """
        metrics_key = self.get_model_metrics_key(model_name)
        metrics = self.redis.hgetall(metrics_key)
        # Convert string values to appropriate types
        result = {}
        for key, value in metrics.items():
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
        return result

    def get_queue_length(self, model_name: str) -> int:
        """Get the number of requests in the queue for a model.

        Args:
            model_name: Name of the model.

        Returns:
            Number of requests in the queue.
        """
        stream_key = self.get_model_stream_key(model_name)
        return self.redis.xlen(stream_key)

    def get_consumer_groups(self, model_name: str) -> List[Dict[str, Any]]:
        """Get consumer groups for a model's stream.

        Args:
            model_name: Name of the model.

        Returns:
            List of consumer group information.
        """
        stream_key = self.get_model_stream_key(model_name)
        try:
            return self.redis.xinfo_groups(stream_key)
        except redis.exceptions.ResponseError:
            # Stream doesn't exist or has no consumer groups
            return []

    def create_consumer_group(self, model_name: str, group_name: str) -> bool:
        """Create a consumer group for a model's stream.

        Args:
            model_name: Name of the model.
            group_name: Name of the consumer group.

        Returns:
            True if the group was created, False otherwise.
        """
        stream_key = self.get_model_stream_key(model_name)
        try:
            self.redis.xgroup_create(
                stream_key, group_name, id="0", mkstream=True)
            return True
        except redis.exceptions.ResponseError:
            # Group already exists
            return False

    def read_group(
        self,
        model_name: str,
        group_name: str,
        consumer_name: str,
        count: int = 10,
        block: int = 0,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Read requests from a consumer group.

        Args:
            model_name: Name of the model.
            group_name: Name of the consumer group.
            consumer_name: Name of the consumer.
            count: Maximum number of requests to get.
            block: Time to block in milliseconds. 0 means no blocking.

        Returns:
            List of (message_id, request_data) tuples.
        """
        stream_key = self.get_model_stream_key(model_name)
        messages = self.redis.xreadgroup(
            group_name,
            consumer_name,
            {stream_key: ">"},
            count=count,
            block=block
        )
        if not messages:
            return []

        result = []
        for stream_name, stream_messages in messages:
            for message_id, data in stream_messages:
                # Convert sampling_params and prompt_token_ids back to Python
                # objects
                if data.get("sampling_params"):
                    data["sampling_params"] = json.loads(
                        data["sampling_params"])
                if data.get("prompt_token_ids"):
                    data["prompt_token_ids"] = json.loads(
                        data["prompt_token_ids"])
                result.append((message_id, data))
        return result

    def ack_group(self, model_name: str, group_name: str,
                  message_id: str) -> int:
        """Acknowledge a message in a consumer group.

        Args:
            model_name: Name of the model.
            group_name: Name of the consumer group.
            message_id: ID of the message in the stream.

        Returns:
            Number of messages acknowledged.
        """
        stream_key = self.get_model_stream_key(model_name)
        return self.redis.xack(stream_key, group_name, message_id)
