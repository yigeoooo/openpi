# 适配一个新机器人:训练与推理完整指南

> 本文是动手指南:从零把一个新机器人接入本项目,跑通训练和推理。
> 阅读前建议先看 [项目架构说明](./architecture_zh.md),理解 transform 管线这一核心概念。
> 全程以官方带注释的 [`libero_policy.py`](../src/openpi/policies/libero_policy.py) 和本项目的 [`alohamini_policy.py`](../src/openpi/policies/alohamini_policy.py) 为模板。

---

## 0. 模型(必读)

模型固定,只认标准格式:**最多 3 路图像 + 1 个 state 向量 + 1 个 prompt**。

适配新机器人 = 写一层**双向翻译**:
- **输入(Inputs)**:你的数据 → 模型标准格式
- **输出(Outputs)**:模型输出 → 你的机器人动作格式

**输入做的每一步,输出都要有对应的反操作**(切维度、改单位、补相机槽),否则训练和推理对不上。模型与训练循环一行都不用改。

---

## 1. 总览

| 步骤 | 做什么 | 文件 |
|------|--------|------|
| ① | 写机器人翻译层 `FooInputs` / `FooOutputs` | 新建 `src/openpi/policies/foo_policy.py` |
| ② | 写数据配置工厂,串起 repack + data_transforms | `src/openpi/training/config.py` |
| ③ | 注册一个训练配置 `TrainConfig` | `src/openpi/training/config.py` 的 `_CONFIGS` |
| ④ | (可选)写单测 / example | `policies/foo_policy.py` + 测试 |
| ⑤ | 算归一化:`compute_norm_stats.py` | 命令行 |
| ⑥ | 训练 + 起推理服务 | 命令行 |

---

## 2. 准备数据集(LeRobot 格式)

本项目主要吃 **LeRobot 数据集**。录制/转换后,确认它具备:

- `meta/info.json`:`fps`、各 feature 的 shape 和 dtype
- `observation.state`:状态向量(维度 = 你的 DoF)
- `action`:动作向量(维度 = 你的 DoF)
- `observation.images.<相机名>`:各路相机(本项目用 video 存)
- `meta/tasks`:`task_index → task 文本` 的映射(如果你要用语言指令)

**检查清单:**
- [ ] state / action 的维度、顺序、单位,和你真机控制接口**完全一致**
- [ ] action 是"要执行的命令",不是"观测到的状态"
- [ ] 每条样本都有合法 `task_index`,且能在 `meta/tasks` 里查到(开了 `prompt_from_task` 后这是硬要求)
- [ ] 去掉开头准备、人工调整、长停顿、失败轨迹、结尾无意义等待

> 可用项目加载器抽样验证:用 `data_loader.create_torch_dataset(...)` 加载后打印几条样本的 `prompt` / `state.shape` / `action.shape` / 相机键,确认都对得上。

---

## 3. ① 写翻译层 `foo_policy.py`

复制 [`libero_policy.py`](../src/openpi/policies/libero_policy.py) 改即可。它注释最全。

### 3.1 FooInputs:你的数据 → 模型标准格式

```python
import dataclasses
import einops
import numpy as np
from openpi import transforms


def _parse_image(image) -> np.ndarray:
    """统一成 uint8 的 (H, W, C)。LeRobot 常存成 float32 的 (C, H, W)。"""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:          # CHW -> HWC
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FooInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # data 的键来自上游 repack(见第 4 节),通常是 images/state/actions/prompt
        base_image = _parse_image(data["images"]["cam_high"])
        left_wrist = _parse_image(data["images"]["cam_left_wrist"])

        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            # 模型固定 3 个图像槽位;没有的相机补零 + mask=False
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,   # π0-FAST 例外,见 libero 注释
            },
        }

        # 动作只在训练时存在
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        # prompt(语言指令)必须透传,且输出键固定叫 "prompt"
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs
```

要点:
- 三个图像槽位 `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb` 是**模型固定的**,键名不能改。
- 缺的相机补零数组并把对应 `image_mask` 设 `False`。
- `state`、`actions`、`prompt` 三个键名也固定。

### 3.2 FooOutputs:模型输出 → 你的机器人动作

```python
@dataclasses.dataclass(frozen=True)
class FooOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 模型输出会 padding 到内部维度,只取你机器人真实的前 N 维
        # N = 你的 action DoF(例如 7 / 16 / 18)
        return {"actions": np.asarray(data["actions"][:, :N])}
```

如有单位换算、范围 clip,在这里做(参考 [`alohamini_policy.py:171`](../src/openpi/policies/alohamini_policy.py) `AlohaMiniOutputs` 里的 `np.clip(...)`)。

### 3.3 (可选)维度对齐

若你的 DoF 与模型内部期望不一致(π0.5 内部按 18D 处理),写一个对齐 transform 在 Inputs 之后插/去虚拟关节。直接参考 [`alohamini_policy.py`](../src/openpi/policies/alohamini_policy.py) 的 `AlignToPi05ActionSpace`(16↔18 互转,插补虚拟关节并固定为 0)。

### 3.4 (可选)example,便于单测

```python
def make_foo_example() -> dict:
    return {
        "observation.state": np.zeros(N, dtype=np.float32),
        "observation.images.cam_high": np.zeros((480, 640, 3), dtype=np.uint8),
        "prompt": "do something",
    }
```

---

## 4. ② 写数据配置工厂(串起 repack + transforms)

在 [`config.py`](../src/openpi/training/config.py) 里加一个 `DataConfigFactory` 子类。可直接参考 `LeRobotAlohaMiniDataConfig`([`config.py:561`](../src/openpi/training/config.py))。它的 `create()` 负责把四组 transform 拼起来。

最小骨架:

```python
@dataclasses.dataclass(frozen=True)
class LeRobotFooDataConfig(DataConfigFactory):
    default_prompt: str | None = None
    use_delta_actions: bool = True

    @override
    def create(self, assets_dirs, model_config) -> DataConfig:
        # repack:把数据集原始列名 → 翻译层期望的中间键
        repack = _transforms.Group(inputs=[
            _transforms.RepackTransform({
                "images": {
                    "cam_high": "observation.images.cam_high",
                    "cam_left_wrist": "observation.images.cam_left_wrist",
                },
                "state": "observation.state",
                "actions": "action",
                "prompt": "prompt",          # ★ 想用语言指令,这行不能少
            })
        ])

        # data_transforms:你的翻译层(进 Inputs / 出 Outputs)
        data_transforms = _transforms.Group(
            inputs=[foo_policy.FooInputs()],
            outputs=[foo_policy.FooOutputs()],
        )

        # (可选)增量动作:哪些维度学相对量,哪些保持绝对
        if self.use_delta_actions:
            mask = [...]   # 长度=动作维度,True=delta
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(mask)],
                outputs=[_transforms.AbsoluteActions(mask)],
            )

        # 标准件,自动生成(InjectDefaultPrompt + Resize + Tokenize + Pad)
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
```

这里要决策的几件事:

| 决策 | 怎么设 |
|------|--------|
| 相机名映射 | repack 里 `"目标槽名": "数据集列名"` |
| 是否用语言指令 | `base_config=DataConfig(prompt_from_task=True)` + repack 含 `"prompt": "prompt"` |
| 增量 vs 绝对动作 | `DeltaActions/AbsoluteActions` 的 mask(如手臂关节学 delta、夹爪/底盘/升降轴绝对) |
| 默认 prompt | `default_prompt`(仅在样本无 prompt 时兜底) |

> **关于 prompt 的两处开关(最易踩坑):**
> 1. `prompt_from_task=True`(通过 `base_config` 传入)
> 2. repack 结构里有 `"prompt": "prompt"`
> 两者缺一,数据集里的 task 描述就进不了训练,只会用默认 prompt。

---

## 5. ③ 注册训练配置 TrainConfig

在 [`config.py`](../src/openpi/training/config.py) 的 `_CONFIGS` 列表里加一项:

```python
TrainConfig(
    name="foo_v1",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10),
    data=LeRobotFooDataConfig(
        repo_id="/path/to/foo_dataset",
        base_config=DataConfig(prompt_from_task=True),   # 用语言指令
        default_prompt="do the task",
        assets=AssetsConfig(
            assets_dir="/path/to/assets",   # norm_stats.json 会放这里
            asset_id="foo_dataset",
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params",   # 从 π05_base 微调
    ),
    num_train_steps=30000,
    batch_size=4,
    num_workers=2,
    wandb_enabled=False,
),
```

> `_CONFIGS` 末尾有唯一性检查:配置名不能重复。

---

## 6. ⑤ 计算归一化统计量(训练前必跑)

```bash
uv run python scripts/compute_norm_stats.py --config-name=foo_v1
```

会在 `assets_dir/asset_id/` 下生成 `norm_stats.json`。**不先跑这步,训练启动会直接报错。**

何时需要重算:
- ✅ 换数据集、改了 state/action 的维度/顺序/单位
- ❌ 只改 prompt / default_prompt(不影响归一化,无需重算)

---

## 7. ⑥ 训练

```bash
uv run python scripts/train.py foo_v1 --exp-name=run1
```

checkpoint 存到 `checkpoints/foo_v1/run1/<step>`,norm_stats 会自动拷进 `checkpoint/assets`(保证推理用同一份归一化)。

开训后立刻看第一个日志:
- data loader 能正常出 batch(说明数据集、维度、相机、prompt 都对得上)
- loss 在下降
- 若 `prompt_from_task=True` 但某些样本 `task_index` 查不到映射,会在这里报错 —— 正好帮你校验数据集完整性

---

## 8. ⑥ 推理 / 起服务

```bash
uv run python scripts/serve_policy.py policy:checkpoint \
  --policy.config=foo_v1 \
  --policy.dir=checkpoints/foo_v1/run1/30000
```

机器人客户端按 `FooInputs` 期望的键发观测(图像 + state + prompt),拿到动作 chunk。

推理时的 transform 装配是训练的**镜像**(正向进、反向出),见 [`policy_config.py:75`](../src/openpi/policies/policy_config.py)。远程推理细节见 [`docs/remote_inference.md`](./remote_inference.md)。

---

## 9. 子任务 prompt + 状态机(长任务推荐做法)

模型每次只输出 `action_horizon` 帧(本项目 10 帧 ≈ 0.33s),它不会"记住整段流程"。对多阶段长任务(如:取货 → 升降轴上升 → 放货 → 下降归位),推荐:

**训练侧**:把数据按阶段切成子任务,每段标不同 `task_index` / task 描述。例如:
```
grasp the bottle on the second shelf
raise the lift to the third shelf
place the bottle on the third shelf
lower the lift to the home position
```
两种录法:(A) 每次只录一个阶段;(B) 录完整任务后按时间点切段重标。关键是**每一帧的 task_index 要对应当前子任务**。

**推理侧**:写一个状态机,根据机器人状态切换喂给模型的 prompt:
```
夹爪闭合/抓到物体     → "raise the lift to the third shelf"
lift 高度到三层       → "place the bottle on the third shelf"
夹爪打开/放货完成     → "lower the lift to the home position"
lift 回到底部         → 结束
```

注意事项:
- 训练和推理的 prompt 字符串要**逐字一致**(模型按语言条件化)。维护一份固定清单,两边都从它取。
- 子任务要**首尾可拼接**:第 N 段的结束状态 = 第 N+1 段的起始状态,否则真机串起来会进入模型没见过的状态。
- 重点把"画面像、动作相反"的阶段(升降轴上升 vs 下降)拆成不同子任务、prompt 写明显不同。
- 各子任务样本数尽量均衡,别一类多一类少。

---

## 10. 常见坑

| 现象 | 原因 / 排查 |
|------|-------------|
| task 描述没进训练,模型只认默认 prompt | `prompt_from_task` 没开,或 repack 没保留 `"prompt": "prompt"`(两者都要) |
| 训练启动报"找不到 norm stats" | 没先跑 `compute_norm_stats.py` |
| 推理动作异常大 / 抖 | Inputs/Outputs 的维度切分、单位、clip 与训练不对称;或推理用的 norm_stats 与训练不一致 |
| 长任务阶段边界混乱 | 用了单条长 prompt;改为子任务 prompt + 状态机 |
| `task_index not found` | 数据集某些帧的 task_index 不在 `meta/tasks` 映射里 |
| 相机数量/命名不符 | Inputs 里把缺失相机补零 + `image_mask=False`;多余相机自行取舍映射到 3 个槽位 |
