import dataclasses
from typing import ClassVar
import einops
import numpy as np

from openpi import transforms


def _api16_to_internal18(x: np.ndarray) -> np.ndarray:
    if x.shape[-1] != 16:
        raise ValueError(f"Expected 16D input for api16_to_internal18, got shape {x.shape}")
    x18 = np.insert(x, 5, 0.0, axis=-1)   
    x18 = np.insert(x18, 12, 0.0, axis=-1)  
    x18[..., 5] = 0.0
    x18[..., 12] = 0.0
    return x18


def _internal18_to_api16(actions18: np.ndarray) -> np.ndarray:
    if actions18.shape[-1] != 18:
        raise ValueError(f"Expected 18D internal actions, got shape {actions18.shape}")

    actions18 = np.asarray(actions18)
    actions18[..., 5] = 0.0
    actions18[..., 12] = 0.0

    h = actions18.shape[-2] if actions18.ndim >= 2 else 1
    a2 = actions18.reshape((h, 18))
    api16 = np.concatenate(
        [
            a2[:, 0:5],    
            a2[:, 6:7],    
            a2[:, 7:12],   
            a2[:, 13:14],  
            a2[:, 14:16],  
            a2[:, 16:17],  
            a2[:, 17:18],  
        ],
        axis=-1,
    )
    return api16


@dataclasses.dataclass(frozen=True)
class AlignToPi05ActionSpace(transforms.DataTransformFn):
    """Align robot state/actions to Pi0.5-required 18D action space.

    - robot_dof=16: insert virtual joints (legacy AlohaMini layout) to expand 16 -> 18
    - robot_dof=18: pass through 18D directly (no virtual joints)
    """

    robot_dof: int = 16

    def _align(self, x: np.ndarray, key: str) -> np.ndarray:
        if self.robot_dof not in (16, 18):
            raise ValueError(f"AlignToPi05ActionSpace: robot_dof must be 16 or 18, got {self.robot_dof}")

        if self.robot_dof == 16:
            if x.shape[-1] == 16:
                y = _api16_to_internal18(x)
            elif x.shape[-1] == 18:
                y = np.asarray(x, dtype=np.float32).copy()
            else:
                raise ValueError(
                    f"AlignToPi05ActionSpace({key}): expected 16D or 18D input for robot_dof=16, got shape={x.shape}"
                )
            # Keep legacy virtual joints fixed at zero.
            y[..., 5] = 0.0
            y[..., 12] = 0.0
            return y

        if self.robot_dof == 18:
            # direct passthrough, no virtual joints.
            if x.shape[-1] != 18:
                raise ValueError(
                    f"AlignToPi05ActionSpace({key}): expected 18D input for robot_dof=18, got shape={x.shape}"
                )
            return np.asarray(x, dtype=np.float32)

        raise ValueError(f"AlignToPi05ActionSpace: unsupported robot_dof={self.robot_dof}")

    def __call__(self, data: dict) -> dict:
        result = dict(data)
        if "state" in result:
            result["state"] = self._align(np.asarray(result["state"], dtype=np.float32), key="state")
        if "actions" in result:
            result["actions"] = self._align(np.asarray(result["actions"], dtype=np.float32), key="actions")
        return result


@dataclasses.dataclass(frozen=True)
class AddVirtualJoint6(transforms.DataTransformFn):
    """Compatibility name, now strictly aligns to 18D internal space."""

    def __call__(self, data: dict) -> dict:
        return AlignToPi05ActionSpace(robot_dof=16)(data)


@dataclasses.dataclass(frozen=True)
class RemoveVirtualJoint6(transforms.DataTransformFn):
    """Compatibility name: convert internal 18D actions to API 16D actions."""

    def __call__(self, data: dict) -> dict:
        result = dict(data)

        if "actions" in result:
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.shape[-1] == 18:
                result["actions"] = _internal18_to_api16(actions)

        return result


@dataclasses.dataclass(frozen=True)
class AlohaMiniInputs(transforms.DataTransformFn):
    bgr_to_rgb: bool = False

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high", "cam_left_wrist", "cam_right_wrist", "cam_low")

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(
                f"Unexpected image keys: {tuple(in_images)}. Expected subset of {self.EXPECTED_CAMERAS}.")

        def convert_image(img):
            img = np.asarray(img)
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            hwc = einops.rearrange(img, "c h w -> h w c")
            if self.bgr_to_rgb and hwc.ndim == 3 and hwc.shape[-1] == 3:
                hwc = hwc[..., ::-1].copy()
            return hwc

        images_dict = {name: convert_image(img)
                       for name, img in in_images.items()}

        base_image = images_dict["cam_high"]
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in images_dict:
                images[dest] = images_dict[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        out = {
            "image": images,
            "image_mask": image_masks,
            "state": np.asarray(data["state"], dtype=np.float32),
        }

        if "actions" in data:
            out["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            out["prompt"] = data["prompt"]

        return out


@dataclasses.dataclass(frozen=True)
class AlohaMiniOutputs(transforms.DataTransformFn):
    internal_dim: int | None = 18
    api_dof: int = 16
    robot_dof: int | None = None

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        internal_dim = int(self.internal_dim or 18)
        output_dof = int(self.robot_dof) if self.robot_dof is not None else int(self.api_dof)
        if internal_dim != 18:
            raise ValueError(f"AlohaMiniOutputs: internal_dim must be 18, got {internal_dim}")
        if output_dof not in (16, 18):
            raise ValueError(f"AlohaMiniOutputs: output dof must be 16 or 18, got {output_dof}")
        if actions.shape[-1] < internal_dim:
            raise ValueError(
                f"AlohaMiniOutputs: actions dimension ({actions.shape[-1]}) is smaller than internal_dim={internal_dim}. "
                f"Expected actions to have at least {internal_dim} dimensions after AbsoluteActions."
            )
        internal = actions[:, :18].copy()

        if output_dof == 16:
            internal[:, 5] = 0.0
            internal[:, 12] = 0.0
            api_actions = _internal18_to_api16(internal)
            api_actions = np.asarray(api_actions, dtype=np.float32)
            api_actions[:, 0:5] = np.clip(api_actions[:, 0:5], -100.0, 100.0)
            api_actions[:, 6:11] = np.clip(api_actions[:, 6:11], -100.0, 100.0)
            api_actions[:, 5] = np.clip(api_actions[:, 5], 0.0, 100.0)
            api_actions[:, 11] = np.clip(api_actions[:, 11], 0.0, 100.0)
            api_actions[:, 12:14] = np.clip(api_actions[:, 12:14], -0.15, 0.15)
            api_actions[:, 15] = np.clip(api_actions[:, 15], 0.0, 0.45)
        if output_dof == 18:
            if internal_dim != 18:
                raise ValueError(
                    "AlohaMiniOutputs: robot/output dof=18 requires internal_dim=18. "
                    "This avoids adding virtual joints in the 18-DoF path."
                )
            api_actions = internal.copy()
            api_actions = np.asarray(api_actions, dtype=np.float32)

        if output_dof not in (16, 18):
            raise ValueError(f"AlohaMiniOutputs: unsupported output_dof={output_dof}")
        
        import logging
        logger = logging.getLogger(__name__)
        if not hasattr(self, "_logged_once"):
            logger.debug(
                f"AlohaMiniOutputs INPUT: actions shape={actions.shape}, "
                f"internal range=[{internal.min():.3f}, {internal.max():.3f}], "
                f"internal mean_abs={np.abs(internal).mean():.3f}, "
                f"internal_dim={internal_dim}, output_dof={output_dof}"
            )
            object.__setattr__(self, "_logged_once", True)
        
        if not hasattr(self, "_logged_output_once"):
            logger.debug(
                f"AlohaMiniOutputs OUTPUT: api_actions shape={api_actions.shape}, "
                f"range=[{api_actions.min():.3f}, {api_actions.max():.3f}], "
                f"mean_abs={np.abs(api_actions).mean():.3f}"
            )
            if np.any(np.abs(api_actions) > 1000.0):
                logger.warning(
                    f"AlohaMiniOutputs: CRITICAL - actions extremely large! "
                    f"max_abs={np.abs(api_actions).max():.3f}"
                )
            elif np.any(np.abs(api_actions) > 100.0):
                logger.warning(
                    f"AlohaMiniOutputs: WARNING - actions very large! "
                    f"max_abs={np.abs(api_actions).max():.3f}"
                )
            object.__setattr__(self, "_logged_output_once", True)
        
        return {"actions": api_actions}
