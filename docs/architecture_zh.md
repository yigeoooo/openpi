# OpenPI 项目架构说明

> 本文帮助快速理解本仓库(基于 π0 / π0.5 的 VLA 训练与推理框架)的整体结构、数据流和关键代码位置。
> 配套文档:[如何适配一个新机器人](./add_new_robot_zh.md)。

---

## 1. 一句话理解

模型(π0 / π0.5)是固定的,它只认一种**标准格式**;每个机器人说自己的"方言"(不同的相机命名、关节维度 DoF、单位)。

> **适配 = 写一层双向翻译。** 把机器人方言翻成模型普通话喂进去(输入),再把模型输出的普通话翻回方言去驱动电机(输出)。

模型固定不变,项目几乎所有复杂度都集中在这层"翻译"(transforms)里。

模型只接受:
- **图像**:固定 3 个槽位 `base_0_rgb`(主视角)、`left_wrist_0_rgb`、`right_wrist_0_rgb`,各带一个 mask
- **状态 state**:一个向量,padding 到模型内部维度
- **提示词 prompt**:文本,tokenize 后输入
- **动作 actions**(仅训练时):padding 到模型内部维度

---

## 2. 目录结构

```
src/openpi/
├── models/                 模型定义(一般不用动)
│   ├── pi0.py / pi0_fast.py    π0 / π0-FAST 模型
│   ├── pi0_config.py           Pi0Config(pi05=True/False, action_horizon, ...)
│   ├── gemma.py / gemma_fast.py 语言主干
│   └── model.py                Observation / Actions 等数据结构、基类
├── policies/               ★ 机器人专属"翻译层"(适配新机器人主要在这里)
│   ├── alohamini_policy.py     AlohaMini2Pro 适配(本项目主力,可作模板)
│   ├── libero_policy.py        官方带注释的最简模板,适合照抄
│   ├── aloha_policy.py / droid_policy.py  其它机器人样例
│   ├── policy_config.py        create_trained_policy():推理时装配 Policy
│   └── policy.py               Policy 类:infer() 跑正向+反向 transform + 采样
├── transforms.py           ★ 所有通用 transform(管线的"心脏")
├── training/
│   ├── config.py           ★ DataConfig / TrainConfig / 所有训练配置在此注册
│   ├── data_loader.py      数据集加载 + transform 串联
│   ├── optimizer.py / weight_loaders.py / checkpoints.py / sharding.py
│   └── ...
└── serving/                推理服务器(websocket / http)

scripts/
├── train.py                训练入口
├── compute_norm_stats.py   计算归一化统计量(训练前必跑)
├── serve_policy.py         启动推理服务
└── train_pytorch.py        PyTorch 版训练
```

带 ★ 的三个文件是理解和改动项目的核心。

---

## 3. Transform 管线(整个项目的心脏)

数据进出模型要经过 **4 组 transform**。训练只走"进";推理"进"完再**镜像反向**走"出"。

推理时的完整装配见 [`policy_config.py`](../src/openpi/policies/policy_config.py)`:75`:

```
输入(robot → model):
  repack.inputs → InjectDefaultPrompt → data_transforms.inputs → Normalize → model_transforms.inputs

输出(model → robot,完全镜像反向):
  model_transforms.outputs → Unnormalize → data_transforms.outputs → repack.outputs
```

### 四组分别做什么

| 组 | 作用 | 谁写 | 代码位置 |
|----|------|------|----------|
| **repack_transforms** | 纯改名:数据集原始列名 → 中间键(如 `observation.images.chest`→`cam_high`,`action`→`actions`,`prompt`→`prompt`) | 配置里写映射 | `transforms.py` `RepackTransform` |
| **data_transforms** | ★机器人专属翻译:Inputs(方言→普通话)、Outputs(普通话→方言),夹 DeltaActions / 维度对齐 | **你写** | `policies/xxx_policy.py` |
| **Normalize / Unnormalize** | 用 `norm_stats.json` 对 state/action 标准化 / 反标准化 | 框架 | `transforms.py` |
| **model_transforms** | 标准件:InjectDefaultPrompt + ResizeImages(224) + TokenizePrompt + PadStatesAndActions | 框架自动生成 | `config.py` `ModelTransformFactory:108` |

### 关键体会

**输入输出严格镜像。** 你在 Inputs 里做的每一步(切维度、改单位、补相机槽),在 Outputs 里都要有对应的反操作,否则训练和推理对不上。

以 AlohaMini 为例:
- `AlohaMiniInputs`([`alohamini_policy.py:114`](../src/openpi/policies/alohamini_policy.py))把 `{images, state, actions, prompt}` 转成模型标准 dict,顺手做图像 CHW→HWC、float→uint8、缺失相机补零 + `image_mask=False`。
- `AlohaMiniOutputs`([`alohamini_policy.py:171`](../src/openpi/policies/alohamini_policy.py))把模型输出动作切回机器人真实 DoF,做 clip / 单位换算。
- `AlignToPi05ActionSpace` 处理 16↔18 维对齐(给老款 16-DoF 插入虚拟关节)。

---

## 4. Prompt(语言指令)是怎么进模型的

这是一个容易踩坑、值得单独讲的链路:

1. **数据集侧**:LeRobot 数据集里每条 episode 有一个 task 描述,对应一个 `task_index`。
2. **取出来**:仅当 `DataConfig.prompt_from_task=True` 时,数据加载器才挂上 `PromptFromLeRobotTask`,把 `task_index` 翻成 prompt 文本写进样本([`data_loader.py:380`](../src/openpi/training/data_loader.py))。
3. **别被丢掉**:`RepackTransform` 只保留结构里列出的 key。所以 repack 映射里必须有 `"prompt": "prompt"`,否则上一步注入的 prompt 在这步被丢弃。
4. **兜底**:`InjectDefaultPrompt` 只在 prompt 缺失时才填 `default_prompt`([`transforms.py:105`](../src/openpi/transforms.py))。
5. **进模型**:`TokenizePrompt` 把文本 token 化送入模型。

> **常见 bug**:只要 ① `prompt_from_task` 没开 或 ② repack 没保留 `"prompt"`,数据集里写的 task 描述就进不了训练,模型只会看到默认 prompt。两者必须同时满足。

---

## 5. 模型在学什么

模型**不是**把整段几十秒轨迹当一个序列来学。它学的是:

```
当前图像 + 当前 state + prompt  →  未来 action_horizon 帧动作
```

`action_horizon` 在 `Pi0Config` 里设(本项目为 10)。30 fps 下 10 帧 ≈ 0.33 秒。

含义:
- 长任务能否做成,取决于**每个阶段的视觉/状态是否可区分**、prompt 是否消歧,而不是 prompt 一句话能否描述完整流程。
- 画面相似但动作相反的阶段(如"升降轴上升" vs "下降归位"),仅靠一条长 prompt 容易混 → 适合拆成子任务 prompt + 推理端状态机。

---

## 6. 端到端代码流程

### 配置入口
所有脚本通过 `_config.cli()` / `get_config(name)` 把**配置名**解析成 `TrainConfig`。所有配置注册在 [`config.py`](../src/openpi/training/config.py) 的 `_CONFIGS` 列表。

### A. 计算归一化统计量(训练前必跑)
[`compute_norm_stats.py`](../scripts/compute_norm_stats.py)

```bash
uv run python scripts/compute_norm_stats.py --config-name=<config>
```

只套 `[repack, data_transforms, RemoveStrings]`(**故意不做 Normalize、不做 model_transforms**),遍历数据集累计 `state`/`actions` 的 RunningStats,写出 `norm_stats.json` 到 assets 目录。和训练共用前半段管线,保证统计量与训练实际喂入的数一致。

> 换数据集、或改了 state/action 维度/单位 → 必须重算。只改 prompt 不影响归一化,无需重算。

### B. 训练
[`train.py` `main`](../scripts/train.py)

```bash
uv run python scripts/train.py <config> --exp-name=<name>
```

1. `create_data_loader(config)` → `config.data.create(...)` 生成 `DataConfig` → `create_torch_dataset` 加载 LeRobot 数据(若 `prompt_from_task` 则挂 `PromptFromLeRobotTask`)→ `transform_dataset` 套 `[repack, data_transforms, Normalize, model_transforms]` → 每个 batch 打包成 `(Observation, Actions)`。
2. `init_train_state`:`model.create()` 建模型,`weight_loader` 灌入 π05_base 预训练权重。
3. 主循环:`train_step` → `model.compute_loss` → 梯度 → 优化器更新 → 每 `save_interval` 存 checkpoint(norm_stats 一并拷进 `checkpoint/assets`)。

### C. 推理服务
[`serve_policy.py`](../scripts/serve_policy.py) → [`create_trained_policy`](../src/openpi/policies/policy_config.py)

```bash
uv run python scripts/serve_policy.py policy:checkpoint \
  --policy.config=<config> --policy.dir=checkpoints/<config>/<exp>/<step>
```

1. 从 checkpoint 载模型 + 从 `checkpoint/assets` 载 norm_stats(确保与训练用同一份归一化)。
2. 组装 `Policy`:`transforms` 正向五件套,`output_transforms` 反向(`policy_config.py:77`)。
3. WebSocket / HTTP server 收到机器人观测 → `policy.infer(obs)` 正向 transform → `model.sample_actions` 采样 → 反向 transform 翻回机器人格式 → 回传。
4. 机器人端按 chunk 反复请求(`action_chunk_broker`);**状态机在这一层切换发给模型的 prompt 字符串**。

---

## 7. 一图总结

```
                    ┌─────────────────────── 配置 TrainConfig (config.py) ──────────────────────┐
                    │  指定:模型 / 机器人翻译层 data_transforms / 数据集 repo_id / 预训练权重    │
                    └──────────────────────────────────────────────────────────────────────────┘
                                   │                          │                         │
                                   ▼                          ▼                         ▼
   compute_norm_stats.py    ──►  训练 train.py         ──►   推理 serve_policy.py
   repack+data_transforms        repack→data→Norm→model      正向 transform → 采样 → 反向 transform
   累计 state/action 统计         学:图+state+prompt→动作      norm_stats 来自 checkpoint
   产出 norm_stats.json          产出 checkpoint              机器人端状态机切 prompt
```

**核心:配置指定"用哪层翻译 + 哪份数据";归一化先算好统计量;训练让模型学"看图读 prompt 出动作";推理把同一套翻译层反着用一遍。** 模型与训练循环本身基本不用碰。

---

## 8. 关键文件速查

| 想做什么 | 看 / 改哪里 |
|----------|-------------|
| 加新机器人翻译层 | `src/openpi/policies/xxx_policy.py`(抄 `libero_policy.py`) |
| 注册数据配置 / 训练配置 | `src/openpi/training/config.py` 的 `DataConfigFactory` 子类与 `_CONFIGS` |
| 通用 transform(改名/归一化/tokenize/delta) | `src/openpi/transforms.py` |
| 数据怎么加载、prompt 怎么进来 | `src/openpi/training/data_loader.py` |
| 推理时怎么装配 Policy | `src/openpi/policies/policy_config.py` |
| 模型结构 / action_horizon | `src/openpi/models/pi0_config.py`、`pi0.py` |
| 归一化原理 | `docs/norm_stats.md` |
| 远程推理 | `docs/remote_inference.md` |
