# AlohaMini2Pro 使用 OpenPI 训练和推理 Pi-0.5

本文档说明如何在服务器上的 `~/project/openpi` 中，使用已经适配 AlohaMini2Pro 的 OpenPI 配置完成 Pi-0.5 微调训练和 HTTP 推理服务部署。

当前 OpenPI 配置名：

```bash
alohamini2pro
```

当前训练数据集路径：

```bash
/home/jingyi.wang/datasets/dataset2026.06.04
```

数据集信息：

- LeRobot v3.0 格式
- 30 fps
- `observation.state` 是 18 维
- `action` 是 18 维
- 三路相机：
  - `observation.images.chest`
  - `observation.images.wrist_left`
  - `observation.images.wrist_right`

OpenPI 配置中的相机映射：

- `observation.images.chest` -> `cam_high` -> `base_0_rgb`
- `observation.images.wrist_left` -> `cam_left_wrist` -> `left_wrist_0_rgb`
- `observation.images.wrist_right` -> `cam_right_wrist` -> `right_wrist_0_rgb`

AlohaMini2Pro 是原生 18 DoF，所以配置中使用：

```python
robot_dof=18
dataset_action_dim=18
```

当前 `alohamini2pro` 训练配置默认冻结 VLM 基座：冻结 PaliGemma 视觉塔和语言/Gemma 主干，只训练 action expert、动作投影层和时间 MLP 等动作相关参数。

## 1. 18 维状态和动作顺序

训练和推理都必须保持下面这个顺序：

```text
arm_left_shoulder_pan.pos
arm_left_shoulder_lift.pos
arm_left_elbow_flex.pos
arm_left_wrist_flex.pos
arm_left_wrist_yaw.pos
arm_left_wrist_roll.pos
arm_left_gripper.pos
arm_right_shoulder_pan.pos
arm_right_shoulder_lift.pos
arm_right_elbow_flex.pos
arm_right_wrist_flex.pos
arm_right_wrist_yaw.pos
arm_right_wrist_roll.pos
arm_right_gripper.pos
x.vel
y.vel
theta.vel
lift_axis.height_mm
```

其中：

- 前 14 维是左右臂关节和夹爪
- 后 4 维是底盘和升降轴
- `x.vel`、`y.vel`、`theta.vel` 是底盘速度
- `lift_axis.height_mm` 是升降轴目标高度

当前配置的 `delta_action_mask` 含义是：

- 双臂非夹爪关节使用 delta action
- 左右夹爪、底盘、升降轴保持 absolute action

## 2. 进入服务器和项目

```bash
ssh jingyi.wang@183.230.224.121 -p 50210
cd ~/project/openpi
```

## 3. 安装 OpenPI 环境

如果服务器还没有安装 `uv`，需要先安装 `uv`。安装后在 `~/project/openpi` 下执行：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

本项目新增的 HTTP 推理服务依赖：

- `fastapi`
- `uvicorn`

它们已经写入 `pyproject.toml`。

## 4. 计算 OpenPI 归一化统计量

注意：LeRobot 数据集自带的 `meta/stats.json` 不能直接替代 OpenPI 训练需要的 `norm_stats.json`。

当前配置会从下面的位置读取 OpenPI norm stats：

```text
/home/jingyi.wang/datasets/dataset2026.06.04/norm_stats.json
```

第一次训练前需要执行：

```bash
cd ~/project/openpi
uv run scripts/compute_norm_stats.py --config-name alohamini2pro
```

执行完成后应能看到：

```text
/home/jingyi.wang/datasets/dataset2026.06.04/norm_stats.json
```

## 5. 启动 Pi-0.5 微调训练

```bash
cd ~/project/openpi
CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run scripts/train.py alohamini2pro \
    --exp-name=alohamini2pro_pi05_0604 \
    --overwrite \
    --fsdp-devices=4 \
    --save_interval=10000 \
    --ema-decay=None
```

可选参数：--ema-decay=None，训练时会优化显存空间，跑不起来可以尝试加上，对应可能导致模型推理没有那么稳定

当前配置使用的 Pi-0.5 base 权重是：

```text
gs://openpi-assets/checkpoints/pi05_base/params
```

OpenPI 会按自身逻辑下载或缓存该权重。

训练输出目录：

```text
~/project/openpi/checkpoints/alohamini2pro/alohamini2pro_pi05_0604/<step>
```

其中 `<step>` 是具体保存步数，例如 `19999` 或其他 checkpoint step。

## 6. 启动 HTTP 推理服务

本项目已经从同级目录的 `AlohaMini` 仓库复制了 HTTP 推理服务代码：

```text
src/openpi/serving/http_policy_server.py
```

并新增了 OpenPI 内部启动入口：

```text
scripts/serve_policy_http.py
```

训练完成后，用下面命令启动 HTTP 推理服务：

```bash
cd ~/project/openpi
uv run scripts/serve_policy_http.py \
  --port 8000 \
  --default-prompt "pickup the rubbish" \
  policy:checkpoint \
  --policy.config=alohamini2pro \
  --policy.dir=checkpoints/alohamini2pro/alohamini2pro_pi05_0604/<step>
```

服务接口：

- `GET /healthz`
- `GET /metadata`
- `POST /infer`

可以先检查服务是否启动：

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/metadata
```

### 6.1 机器人侧连接 HTTP 推理服务

机器人侧运行 LeRobot/AlohaMini2Pro 推理脚本时，`--server_url` 必须填写机器人侧机器能够访问到的 HTTP 地址。服务端启动成功后，如果机器人侧请求真的到达 OpenPI 服务端，服务端终端会看到 `GET /metadata` 和 `POST /infer` 访问日志。

#### 方式 A：服务器 8000 端口已开放，直接访问

如果云服务器或内网服务器已经把 `8000` 端口开放给机器人侧机器，并且防火墙、安全组、NAT 都允许访问，可以直接使用服务器 IP：

```bash
curl http://<server-ip>:8000/healthz
curl http://<server-ip>:8000/metadata
```

确认能返回后，在机器人侧运行：

```bash
cd ~/project/lerobot_alohamini
python3 examples/alohamini/evaluate_bi_http.py \
  --server_url http://<server-ip>:8000 \
  --remote_ip <robot-host-ip> \
  --robot_model alohamini2pro \
  --robot_dof 18 \
  --task_description "pickup the rubbish" \
  --no_save
```

如果 `curl http://<server-ip>:8000/healthz` 失败，或者返回 `Empty reply from server`，说明公网/内网端口没有正常打到 Uvicorn 进程，不要继续用这个地址跑推理，先改用下面的 SSH 转发方式。

#### 方式 B：服务器 8000 端口未开放，使用 SSH 本地转发

如果服务器只开放 SSH 端口，例如：

```text
ssh jingyi.wang@183.230.224.121 -p 50210
```

但没有开放 HTTP `8000` 端口，可以在机器人侧机器新开一个终端，建立本地端口转发：

```bash
ssh -p 50210 -L 18000:127.0.0.1:8000 jingyi.wang@183.230.224.121 -N
```

保持这个终端不要关闭。它会把机器人侧机器的 `127.0.0.1:18000` 转发到服务器本机的 `127.0.0.1:8000`。

然后在机器人侧另一个终端检查：

```bash
curl http://127.0.0.1:18000/healthz
curl http://127.0.0.1:18000/metadata
```

确认正常后，机器人侧推理脚本使用本地转发地址：

```bash
cd ~/project/lerobot_alohamini
python3 examples/alohamini/evaluate_bi_http.py \
  --server_url http://127.0.0.1:18000 \
  --remote_ip <robot-host-ip> \
  --robot_model alohamini2pro \
  --robot_dof 18 \
  --task_description "pickup the rubbish" \
  --no_save
```

注意：

- SSH 转发命令必须在运行机器人侧推理脚本的同一台机器上执行。
- 如果本地 `18000` 端口被占用，可以换成其他端口，例如 `19000`，并同步修改 `--server_url http://127.0.0.1:19000`。
- 使用 SSH 转发时，OpenPI 服务端仍然只需要监听服务器本机的 `0.0.0.0:8000` 或 `127.0.0.1:8000`，机器人侧不需要直接访问服务器公网 `8000` 端口。
- 如果服务端终端始终没有出现 `GET /metadata` 或 `POST /infer`，说明机器人侧请求还没有到达 OpenPI HTTP 服务。

## 7. 推理请求格式

HTTP 服务期望收到 OpenPI 风格 observation：

```python
observation = {
    "images": {
        "cam_high": chest_image_chw,
        "cam_left_wrist": wrist_left_image_chw,
        "cam_right_wrist": wrist_right_image_chw,
    },
    "state": state_18d,
    "prompt": "pickup the rubbish",
}
```

要求：

- `state_18d` 必须是 18 维，并且顺序必须和本文档第 1 节一致
- `cam_high` 对应 AlohaMini2Pro 的 `chest` 相机
- 图片可以是数组，也可以是 `http_policy_server.py` 支持的编码 payload

服务返回：

- `action`：预测 action chunk 的第一步，18 维
- `actions`：完整 action chunk，形状约为 `[action_horizon, 18]`
- `server_timing`：服务端耗时信息
