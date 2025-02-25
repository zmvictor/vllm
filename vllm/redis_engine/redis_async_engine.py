import asyncio
import time
import uuid
from typing import List, Optional, Union

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.redis.redis_client import RedisClient


class RedisAsyncLLMEngine:
    """Asynchronous LLM engine that integrates with Redis for request queuing.

    This class wraps the AsyncLLMEngine and adds Redis integration for
    distributed request handling. It can process requests from Redis streams
    and directly through the API.
    """

    def __init__(
        self,
        engine: AsyncLLMEngine,
        model_name: str,
        redis_client: RedisClient,
        consumer_group: str = "vllm_workers",
        consumer_name: Optional[str] = None,
        poll_interval: float = 0.1,
        batch_size: int = 10,
    ):
        """Initialize the Redis async LLM engine.

        Args:
            engine: The AsyncLLMEngine instance.
            model_name: Name of the model being served.
            redis_client: RedisClient instance for Redis communication.
            consumer_group: Name of the Redis consumer group.
            consumer_name: Name of this consumer. If None, a UUID will be used.
            poll_interval: Interval in seconds to poll Redis for new requests.
            batch_size: Maximum number of requests to process in a batch.
        """
        self.engine = engine
        self.model_name = model_name
        self.redis_client = redis_client
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name or str(uuid.uuid4())
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.running = False
        self.consumer_task = None
        self._setup_consumer_group()

    def _setup_consumer_group(self) -> None:
        """Set up the Redis consumer group for this model."""
        self.redis_client.create_consumer_group(
            self.model_name, self.consumer_group
        )

    async def start(self) -> None:
        """Start the Redis consumer task."""
        if self.running:
            return
        self.running = True
        self.consumer_task = asyncio.create_task(self._consume_requests())

    async def stop(self) -> None:
        """Stop the Redis consumer task."""
        if not self.running:
            return
        self.running = False
        if self.consumer_task:
            self.consumer_task.cancel()
            try:
                await self.consumer_task
            except asyncio.CancelledError:
                pass
            self.consumer_task = None

    async def _consume_requests(self) -> None:
        """Consume requests from Redis and process them."""
        while self.running:
            try:
                # Read requests from Redis
                messages = self.redis_client.read_group(
                    self.model_name,
                    self.consumer_group,
                    self.consumer_name,
                    count=self.batch_size,
                    block=int(self.poll_interval * 1000),
                )

                if not messages:
                    await asyncio.sleep(self.poll_interval)
                    continue

                # Process each request
                for message_id, data in messages:
                    request_id = data.get("request_id")
                    prompt = data.get("prompt")
                    sampling_params = data.get("sampling_params", {})
                    prompt_token_ids = data.get("prompt_token_ids")

                    # Convert sampling_params dict to SamplingParams object
                    if isinstance(sampling_params, dict):
                        sampling_params = SamplingParams(**sampling_params)

                    # Process the request
                    try:
                        await self.engine.generate(
                            prompt=prompt,
                            sampling_params=sampling_params,
                            request_id=request_id,
                            prompt_token_ids=prompt_token_ids,
                        )

                        # Acknowledge the message after successful processing
                        self.redis_client.ack_group(
                            self.model_name,
                            self.consumer_group,
                            message_id,
                        )
                    except Exception as e:
                        # Log the error but continue processing other requests
                        print(f"Error processing request {request_id}: {e}")

            except Exception as e:
                # Log the error but continue the consumer loop
                print(f"Error in Redis consumer: {e}")
                await asyncio.sleep(self.poll_interval)

    async def generate(
        self,
        prompt: Optional[str] = None,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        use_redis: bool = False,
    ) -> Union[str, RequestOutput]:
        """Generate text from the model.

        Args:
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            request_id: The unique ID of the request.
            prompt_token_ids: The token IDs of the prompt. If None, the prompt
                will be tokenized using the model's tokenizer.
            use_redis: Whether to queue the request in Redis instead of
                processing it directly.

        Returns:
            If use_redis=True, returns the request_id.
            If use_redis=False, returns the RequestOutput from the engine.
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        if request_id is None:
            request_id = str(uuid.uuid4())

        if use_redis:
            # Queue the request in Redis
            # Convert sampling_params to a dict for serialization
            params_dict = {
                "temperature": sampling_params.temperature,
                "top_p": sampling_params.top_p,
                "max_tokens": sampling_params.max_tokens,
            }
            self.redis_client.add_request(
                model_name=self.model_name,
                request_id=request_id,
                prompt=prompt,
                sampling_params=params_dict,
                prompt_token_ids=prompt_token_ids,
            )
            return request_id
        else:
            # Process the request directly
            return await self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
            )

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Args:
            request_id: The ID of the request to abort.
        """
        await self.engine.abort(request_id)

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        model_name: str,
        redis_client: RedisClient,
        **kwargs,
    ) -> "RedisAsyncLLMEngine":
        """Create a RedisAsyncLLMEngine from engine arguments.

        Args:
            engine_args: The arguments for creating the AsyncLLMEngine.
            model_name: Name of the model being served.
            redis_client: RedisClient instance for Redis communication.
            **kwargs: Additional arguments for RedisAsyncLLMEngine.

        Returns:
            A RedisAsyncLLMEngine instance.
        """
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        return cls(
            engine=engine,
            model_name=model_name,
            redis_client=redis_client,
            **kwargs,
        )

    def update_metrics(self) -> None:
        """Update metrics in Redis."""
        # Get metrics from the engine
        metrics = {
            "queue_length": 0,  # Will be populated by the autoscaler
            "running_requests": 0,  # Will be populated by the autoscaler
            "last_updated": time.time(),
        }

        # Update metrics in Redis
        self.redis_client.update_metrics(self.model_name, metrics)
