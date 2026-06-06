"""HTTP REST API server for openpi policy inference.

This module provides a FastAPI-based HTTP server for running policy inference,
as an alternative to the WebSocket-based server. This makes it easier to integrate
with standard HTTP clients and tools.
"""

import base64
import io
import logging
import time
from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy
from PIL import Image
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    """Request model for policy inference.

    We support two payload formats for backward/forward compatibility:

    1) OpenPI HTTP format:
       {
         "observation": { ... },
         "noise": [[...], ...]
       }

    2) LeRobot RemotePolicy-compatible format:
       {
         "obs": { ... },   # encoded via _encode_payload() (supports __image_jpeg__ / __ndarray__)
         "task": "..." | null,
         "noise": [[...], ...]
       }
    """

    observation: dict[str, Any] | None = Field(default=None, description="OpenPI observation dict")
    obs: dict[str, Any] | None = Field(default=None, description="LeRobot-style observation dict (encoded)")
    task: str | None = Field(default=None, description="Optional task string (mapped to 'prompt' if missing)")
    noise: list[list[float]] | None = Field(default=None, description="Optional noise array for action sampling")


class InferenceResponse(BaseModel):
    # For LeRobot RemotePolicy compatibility (single step action vector).
    action: list[float] = Field(..., description="First action (one-step) from the predicted action chunk")
    # Full action chunk for callers that want it.
    actions: list[list[float]] = Field(..., description="Predicted action chunk")
    state: list[float] | None = Field(default=None, description="State information if available")
    server_timing: dict[str, float] = Field(..., description="Server timing information in milliseconds")


class HealthResponse(BaseModel):
    status: str = "ok"
    metadata: dict[str, Any] | None = None


def _decode_base64_image(image_data: dict[str, Any]) -> np.ndarray:
    if isinstance(image_data, dict) and "data" in image_data:
        data = image_data["data"]
        # The client may provide a format hint; PIL can infer from bytes, so we don't need it.
        _ = image_data.get("format", "png")
        # Remove data URL prefix if present
        if "," in data:
            data = data.split(",", 1)[1]
        image_bytes = base64.b64decode(data)
        image = Image.open(io.BytesIO(image_bytes))
        return np.array(image)
    raise ValueError("Invalid image data format. Expected dict with 'data' and optional 'format' keys.")


def _decode_jpeg_b64(b64_data: str) -> np.ndarray:
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
    image_bytes = base64.b64decode(b64_data)
    image = Image.open(io.BytesIO(image_bytes))
    arr = np.asarray(image)
    # Convert HWC -> CHW if it looks like a color image.
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        arr = np.transpose(arr, (2, 0, 1))
    return arr


def _decode_remote_payload(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj.get("__image_jpeg__") is True and "b64" in obj:
            return _decode_jpeg_b64(obj["b64"])
        if obj.get("__ndarray__") is True and {"dtype", "shape", "data"} <= set(obj):
            dtype = np.dtype(obj["dtype"])
            shape = tuple(int(x) for x in obj["shape"])
            data = np.asarray(obj["data"], dtype=dtype)
            return data.reshape(shape)
        return {k: _decode_remote_payload(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_remote_payload(v) for v in obj]
    return obj


def _prepare_observation(obs: dict[str, Any]) -> dict[str, Any]:
    def _normalize(x: Any) -> Any:
        if isinstance(x, dict) and "data" in x:
            # OpenPI HTTP image payload: {"data": "...", "format": "..."}
            try:
                return _decode_base64_image(x)
            except Exception:
                return {k: _normalize(v) for k, v in x.items()}
        if isinstance(x, dict):
            return {k: _normalize(v) for k, v in x.items()}
        if isinstance(x, list):
            try:
                arr = np.asarray(x)
                # If this is an image (likely nested in images dict), ensure it's float32
                # Check if it looks like an image: 3D array with reasonable image dimensions
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[1] > 10 and arr.shape[2] > 10:
                    # Likely an image in [C, H, W] format
                    if arr.dtype in (np.int64, np.int32, np.uint8):
                        # Convert integer images to float32 in [0, 1] range
                        arr = arr.astype(np.float32) / 255.0
                    elif arr.dtype == np.float64:
                        # Convert float64 to float32
                        arr = arr.astype(np.float32)
                return arr
            except Exception:
                return [_normalize(v) for v in x]
        return x

    obs_normalized = _normalize(obs)

    # Convert legacy LeRobot-style dicts to OpenPI format if needed.
    obs_converted = _convert_lerobot_to_openpi_format(obs_normalized)

    # For OpenPI inputs, ensure images are CHW as expected by Aloha/AlohaMini input adapters.
    if isinstance(obs_converted, dict) and isinstance(obs_converted.get("images"), dict):
        obs_converted["images"] = {k: _ensure_chw_format(v) for k, v in obs_converted["images"].items()}

    return obs_converted


def _convert_lerobot_to_openpi_format(obs: dict[str, Any]) -> dict[str, Any]:
    
    if "images" in obs and isinstance(obs["images"], dict):
        return obs
    
    # Try to extract images from LeRobot format
    images = {}
    
    # Check for observation.images.* format
    obs_images = obs.get("observation.images", {})
    if isinstance(obs_images, dict):
        # Direct mapping if keys match
        for key in ["cam_high", "cam_left_wrist", "cam_right_wrist", "cam_low"]:
            if key in obs_images:
                img = obs_images[key]
                # Ensure image is in [C, H, W] format
                images[key] = _ensure_chw_format(img)
    
    # Check for observation/images/* format (with slashes or dots)
    for key in list(obs.keys()):
        if key.startswith("observation.images."):
            cam_name = key.replace("observation.images.", "")
            if cam_name in ["cam_high", "cam_left_wrist", "cam_right_wrist", "cam_low"]:
                images[cam_name] = _ensure_chw_format(obs[key])
        elif key.startswith("observation/images/"):
            cam_name = key.replace("observation/images/", "")
            if cam_name in ["cam_high", "cam_left_wrist", "cam_right_wrist", "cam_low"]:
                images[cam_name] = _ensure_chw_format(obs[key])
    
    # Check for single observation/image (map to cam_high)
    if "observation/image" in obs and "cam_high" not in images:
        images["cam_high"] = _ensure_chw_format(obs["observation/image"])
    elif "observation.image" in obs and "cam_high" not in images:
        images["cam_high"] = _ensure_chw_format(obs["observation.image"])
    
    # Extract state
    state = None
    if "observation/state" in obs:
        state = obs["observation/state"]
    elif "observation.state" in obs:
        state = obs["observation.state"]
    elif "state" in obs:
        state = obs["state"]
    
    # Build output dict
    result = {}
    if images:
        result["images"] = images
    if state is not None:
        result["state"] = state
    
    # Copy other keys (prompt, etc.)
    for key, value in obs.items():
        if not (key.startswith("observation.") or key.startswith("observation/")):
            if key not in result:
                result[key] = value
    
    # If we couldn't convert, return original (may work for other policy types)
    if not images and not state:
        return obs
    
    return result


def _ensure_chw_format(img: np.ndarray) -> np.ndarray:
    
    img = np.asarray(img)
    
    # If already 2D (grayscale) or 1D, return as-is (will be handled by policy)
    if img.ndim < 3:
        return img
    
    # Check if it's [H, W, C] format (last dim is small, likely channels)
    if img.ndim == 3:
        h, w, c = img.shape
        # If last dimension is small (<= 4), likely [H, W, C]
        if c <= 4 and h > c and w > c:
            # Convert [H, W, C] -> [C, H, W]
            img = np.transpose(img, (2, 0, 1))
        # Otherwise assume it's already [C, H, W]
    
    return img


class HTTPPolicyServer:
    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict[str, Any] | None = None,
        supported_state_dims: tuple[int, ...] = (16, 18),
    ) -> None:
        """Initialize the HTTP policy server.

        Args:
            policy: The policy to serve.
            host: Host to bind to (default: "0.0.0.0").
            port: Port to bind to (default: 8000).
            metadata: Optional metadata to include in responses.
            supported_state_dims: Accepted observation.state dimensions.
        """
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._supported_state_dims = tuple(sorted(set(int(x) for x in supported_state_dims)))

        self.app = FastAPI(
            title="OpenPI Policy Server",
            description="HTTP REST API server for openpi policy inference",
            version="1.0.0",
        )

        # Register routes
        self._register_routes()

    def _register_routes(self) -> None:
        """Register API routes."""

        @self.app.get("/healthz", response_model=HealthResponse)
        async def health_check() -> HealthResponse:
            """Health check endpoint."""
            return HealthResponse(status="ok", metadata=self._metadata)

        @self.app.get("/metadata", response_model=dict)
        async def get_metadata() -> dict[str, Any]:
            """Get policy metadata."""
            return self._metadata

        @self.app.post("/infer", response_model=InferenceResponse)
        async def infer(request: InferenceRequest) -> InferenceResponse:
            try:
                start_time = time.monotonic()

                # OpenPI format is the primary input contract.
                # `obs` is supported for legacy LeRobot RemotePolicy compatibility.
                obs_in = request.observation if request.observation is not None else request.obs
                if obs_in is None:
                    raise HTTPException(status_code=422, detail="Missing required field: 'obs' or 'observation'")

                # Decode LeRobot payload wrappers if present, then normalize.
                obs_decoded = _decode_remote_payload(obs_in)
                if not isinstance(obs_decoded, dict):
                    raise HTTPException(status_code=422, detail="'obs'/'observation' must be a JSON object (dict)")

                obs = _prepare_observation(obs_decoded)

                # Map task -> prompt if prompt missing (RemotePolicy uses 'task' separately).
                if request.task and "prompt" not in obs:
                    obs["prompt"] = request.task

                # ---- Validate OpenPI observation contract ----
                images = obs.get("images")
                if not isinstance(images, dict):
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "OpenPI observation must include 'images' dict. "
                            "Expected keys like images.cam_high / images.cam_left_wrist / images.cam_right_wrist."
                        ),
                    )
                if "cam_high" not in images:
                    raise HTTPException(
                        status_code=422,
                        detail=f"OpenPI observation.images must include 'cam_high'. Got: {sorted(images.keys())}",
                    )
                if "state" not in obs:
                    raise HTTPException(
                        status_code=422,
                        detail="OpenPI observation must include 'state' (1D vector, typically 16 dims for pick_up_merged).",
                    )
                state = np.asarray(obs["state"])
                if state.ndim != 1:
                    raise HTTPException(status_code=422, detail=f"'state' must be a 1D vector. Got shape={state.shape}.")
                if state.shape[0] not in self._supported_state_dims:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"'state' dim must be one of {self._supported_state_dims}. "
                            f"Got {state.shape[0]}."
                        ),
                    )

                # Convert noise if provided
                noise = None
                if request.noise is not None:
                    noise = np.array(request.noise)

                # Run inference
                infer_start = time.monotonic()
                result = self._policy.infer(obs, noise=noise)
                infer_time = (time.monotonic() - infer_start) * 1000  # Convert to milliseconds

                # Extract actions and state
                actions = result.get("actions", [])
                state = result.get("state")

                # Convert numpy arrays to lists for JSON serialization
                if isinstance(actions, np.ndarray):
                    actions = actions.tolist()
                if state is not None and isinstance(state, np.ndarray):
                    state = state.tolist()

                # Single-step action vector for LeRobot RemotePolicy compatibility.
                try:
                    action0 = actions[0]
                    if isinstance(action0, np.ndarray):
                        action0 = action0.tolist()
                    # Ensure list[float]
                    action = list(action0)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to extract 1-step action from actions: {e}") from e

                total_time = (time.monotonic() - start_time) * 1000

                # Get server timing from result if available
                server_timing = result.get("server_timing", {})
                server_timing["infer_ms"] = infer_time
                server_timing["total_ms"] = total_time

                return InferenceResponse(
                    action=action,
                    actions=actions,
                    state=state,
                    server_timing=server_timing,
                )

            except Exception as e:
                logger.exception("Error during inference")
                raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

        @self.app.get("/")
        async def root() -> dict[str, str]:
            """Root endpoint with API information."""
            return {
                "message": "OpenPI Policy Server",
                "version": "1.0.0",
                "endpoints": {
                    "health": "/healthz",
                    "metadata": "/metadata",
                    "inference": "/infer",
                },
            }

    def serve(self, **kwargs: Any) -> None:        
        uvicorn.run(
            self.app,
            host=self._host,
            port=self._port,
            log_level="info",
            **kwargs,
        )

    def serve_forever(self) -> None:
        """Start the server and run forever."""
        self.serve()
