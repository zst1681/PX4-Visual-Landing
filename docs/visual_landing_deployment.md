# 视觉精准降落项目环境与部署流程

本文档按当前仓库实现整理，目标是把项目推到个人 GitHub 后，可以在一台新的 Ubuntu 设备上快速复现 PX4 SITL + Gazebo + MAVROS + ArUco 视觉降落流程。

## 1. 当前项目组成

核心仓库是 `PX4-Visual-Landing`，在原 PX4 基础上增加了以下视觉降落相关内容。

注意：GitHub 仓库根目录就是本机 `PX4_Firmware` 目录里的所有子文件和子目录，不会再包含一层 `PX4_Firmware/` 文件夹。本文后续用 `PX4_DIR` 表示“本机 PX4 仓库根目录”；新设备推荐克隆到 `$HOME/PX4-Visual-Landing`，当前机器如果仍使用旧目录名，也可以把 `PX4_DIR` 设为 `$HOME/PX4_Firmware`。

- `launch/aruco_search_and_land_demo.launch`：一键启动 Gazebo、PX4 SITL、MAVROS、ArUco 检测和搜索降落控制。
- `launch/aruco_detect_and_search.launch`：启动检测节点和降落控制节点。
- `launch/mavros_posix_sitl_aruco_project.launch`：启动 PX4 SITL、Gazebo 和 MAVROS。
- `scripts/aruco_multi_marker_det.py`：基于 OpenCV ArUco 的多 marker 检测节点，输出 `/aruco/pose`。
- `scripts/aruco_search_and_detect.py`：搜索、对准、下降、切换 AUTO.LAND 的主控制节点。
- `scripts/benchmark_aruco_landing.py`、`scripts/benchmark_pid_groups.py`：批量测试和指标统计。
- `config/*.yaml`、`config/*.json`：相机内参、marker 布局、PID 参数组。
- `Tools/sitl_gazebo` 子模块：包含本项目新增/修改的 ArUco 世界和模型，必须作为子模块单独提交到你自己的 Gazebo fork。

当前项目推荐基线：

- Ubuntu 20.04
- ROS Noetic
- Gazebo 11
- Python 3
- OpenCV 4.2，且 Python 里必须有 `cv2.aruco`
- PX4 SITL 目标：`px4_sitl_default`

## 2. 必需依赖

系统和编译工具：

```bash
sudo apt update
sudo apt install -y git curl wget gnupg lsb-release build-essential cmake ninja-build python3 python3-dev python3-pip python3-setuptools python3-wheel
```

PX4 依赖由仓库自带脚本安装。只做仿真可跳过 NuttX 交叉编译工具链：

```bash
export PX4_DIR=$HOME/PX4-Visual-Landing
cd "$PX4_DIR"
bash Tools/setup/ubuntu.sh --no-nuttx
```

如果后续还要烧录真实飞控固件，去掉 `--no-nuttx`。

ROS 和 MAVROS 依赖：

如果是全新 Ubuntu 20.04，先添加 ROS Noetic 软件源：

```bash
sudo mkdir -p /etc/apt/keyrings
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo gpg --dearmor -o /etc/apt/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros1.list
sudo apt update
```

安装 ROS、MAVROS、Gazebo/ROS 桥接和消息包：

```bash
sudo apt install -y \
  ros-noetic-desktop-full \
  ros-noetic-mavros ros-noetic-mavros-extras \
  ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control \
  ros-noetic-cv-bridge ros-noetic-image-transport \
  ros-noetic-tf ros-noetic-tf2-ros ros-noetic-dynamic-reconfigure \
  ros-noetic-geometry-msgs ros-noetic-sensor-msgs ros-noetic-nav-msgs \
  ros-noetic-std-msgs ros-noetic-visualization-msgs \
  python3-rosdep python3-catkin-tools python3-vcstool
```

初始化 `rosdep`，如果提示已经初始化，可以忽略该提示：

```bash
sudo rosdep init
rosdep update
```

MAVROS 地理数据：

```bash
sudo /opt/ros/noetic/lib/mavros/install_geographiclib_datasets.sh
```

OpenCV/ArUco：

```bash
sudo apt install -y python3-opencv libopencv-dev libopencv-contrib-dev
python3 -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'aruco'))"
```

上面最后一行应输出 `True`。如果不是，说明当前 Python OpenCV 没有 contrib/aruco 模块，需要重新安装带 contrib 的 OpenCV。

Python 依赖：

```bash
export PX4_DIR=$HOME/PX4-Visual-Landing
cd "$PX4_DIR"
python3 -m pip install --user -r Tools/setup/requirements.txt
```

## 3. 外部 catkin 工作空间

当前 `scripts/setup_aruco_runtime.bash` 支持以下环境变量：

- `PX4_DIR`：本机 PX4 仓库根目录，默认是脚本所在仓库；这只是本机路径名，不代表 GitHub 仓库里有 `PX4_Firmware/` 这一层目录。
- `ARUCO_WS`：外部 ArUco 工作空间，默认 `$HOME/ros_gazebo_px4_sim_ws-master`。
- `GAZEBO_WS` 或 `CATKIN_WS`：可选 Gazebo overlay 工作空间，默认 `$HOME/catkin_ws`。
- `XTDRONE_MODELS`：可选 XTDrone 模型目录，默认 `$HOME/XTDrone/sitl_config/models`。
- `PX4_ARUCO_HOME`：运行时临时 HOME，默认 `/tmp/px4_aruco_home`。
- `ROS_DISTRO`：默认 `noetic`。

本项目主流程已经把外部 workspace 改成可选 source；如果你只运行 `aruco_search_and_land_demo.launch`，主要依赖在当前 PX4 仓库和 `Tools/sitl_gazebo` 子模块中。若要保留旧版 `maxi_aruco_det_pkg`、`aruco_ros` 或外部模型，建议把当前机器上的 ArUco catkin 工作空间单独推成一个 GitHub 仓库，部署时默认克隆到 `$HOME/ros_gazebo_px4_sim_ws-master`。

外部工作空间新设备部署示例：

```bash
git clone git@github.com:zst1681/ros_gazebo_px4_sim_ws.git ~/ros_gazebo_px4_sim_ws-master
cd ~/ros_gazebo_px4_sim_ws-master
rosdep install --from-paths src --ignore-src -r -y
catkin build
```

如果你保留 `~/catkin_ws` 作为 Gazebo overlay：

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
catkin build
```

## 4. 个人 GitHub 上传流程

下面按你的 GitHub 信息直接写：

- GitHub 用户名：`zst1681`
- Git 提交邮箱：`z2078543324@163.com`

### 4.0 第一次使用 GitHub 的准备

先在本机配置 Git 身份：

```bash
git config --global user.name "zst1681"
git config --global user.email "z2078543324@163.com"
git config --global init.defaultBranch main
git config --global --list | grep -E "user.name|user.email|init.defaultBranch"
```

建议用 SSH 上传代码。先生成 SSH key：

```bash
ssh-keygen -t ed25519 -C "z2078543324@163.com"
```

一路回车即可。然后启动 ssh-agent 并加入私钥：

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

复制 `cat` 输出的整行公钥，打开 GitHub 网页：

1. 右上角头像 -> `Settings`
2. 左侧 `SSH and GPG keys`
3. `New SSH key`
4. Title 可填 `ubuntu-px4`
5. Key 粘贴刚才的 `id_ed25519.pub` 内容
6. 点击 `Add SSH key`

测试 SSH 是否成功：

```bash
ssh -T git@github.com
```

正常会看到类似 `Hi zst1681! You've successfully authenticated`。如果提示 `Permission denied (publickey)`，说明 SSH key 没有加到 GitHub，或 `ssh-add` 没成功。

然后在 GitHub 网页新建仓库。建议先建这两个：

- `PX4-Visual-Landing`
- `PX4-SITL_gazebo-Visual-Landing`

如果也要保存外部 ArUco ROS 工作空间，再建第三个：

- `ros_gazebo_px4_sim_ws`

新建仓库时不要勾选 `Add a README file`、`.gitignore`、`license`，保持空仓库，方便直接推送本地已有项目。

建议至少维护两个仓库：

- `PX4-Visual-Landing`：当前 PX4 项目仓库，仓库根目录直接是 PX4 文件树。
- `PX4-SITL_gazebo-Visual-Landing`：`Tools/sitl_gazebo` 子模块 fork，因为 ArUco 世界和模型在子模块内。

可选第三个仓库：

- `ros_gazebo_px4_sim_ws`：外部 catkin 工作空间，保留 `src/` 和 README，不提交 `build/`、`devel/`。

### 4.1 推送 Gazebo 子模块

先在 GitHub 创建 `PX4-SITL_gazebo-Visual-Landing`。然后：

```bash
cd /path/to/your/local-px4-repo
export PX4_DIR=$(pwd)
cd Tools/sitl_gazebo
git checkout -b visual-landing-gazebo
git remote rename origin upstream
git remote add origin git@github.com:zst1681/PX4-SITL_gazebo-Visual-Landing.git
git add models/aruco_marker models/aruco_marker_6x6_1000_31_plane models/aruco_nested_board models/iris_down_monocular_cam models/monocular_camera worlds/aruco_landing_demo.world worlds/aruco_search_demo.world worlds/aruco_single_marker_demo.world worlds/empty_aruco.world models/iris_fpv_cam/iris_fpv_cam.sdf
git commit -m "Add ArUco landing Gazebo worlds and models"
git push -u origin visual-landing-gazebo
```

如果还需要 `models/kinect_self`、`worlds/typhoon_h480.world` 或其他已修改模型，也在子模块里一并 `git add`。

注意：`git push -u origin visual-landing-gazebo` 和 `cd "$PX4_DIR"` 是两条命令，必须分两行执行。若 `git commit` 提示没有暂存内容，并且 `git log --oneline -1` 已经能看到 `Add ArUco landing Gazebo worlds and models`，说明这一步已经完成。

### 4.2 更新 PX4 主仓库的子模块地址

这一节必须在主仓库根目录 `$PX4_DIR` 下执行，不要在 `$PX4_DIR/Tools/sitl_gazebo` 子模块目录里执行。

```bash
cd "$PX4_DIR"
git config -f .gitmodules submodule.Tools/sitl_gazebo.url git@github.com:zst1681/PX4-SITL_gazebo-Visual-Landing.git
git config -f .gitmodules submodule.Tools/sitl_gazebo.branch visual-landing-gazebo
git submodule sync Tools/sitl_gazebo
git add .gitmodules Tools/sitl_gazebo
```

### 4.3 推送 PX4 主仓库

在 GitHub 创建 `PX4-Visual-Landing`。然后：

```bash
cd /path/to/your/local-px4-repo
export PX4_DIR=$(pwd)
git checkout -b visual-landing
git remote rename origin upstream
git remote add origin git@github.com:zst1681/PX4-Visual-Landing.git
git add .gitignore docs/visual_landing_deployment.md config launch scripts ROMFS/px4fmu_common/init.d-posix/rcS ROMFS/px4fmu_common/init.d-posix/px4-rc.mavlink .gitmodules Tools/sitl_gazebo
git status
git commit -m "Add ArUco visual landing SITL workflow"
git push -u origin visual-landing
```

不要提交这些运行产物：

- `build/`
- `devel/`
- `.catkin_tools/`
- `generated/`
- `logs/`
- `scripts/__pycache__/`

### 4.4 推送外部 ArUco 工作空间

```bash
cd ~/ros_gazebo_px4_sim_ws-master
git init
git remote add origin git@github.com:zst1681/ros_gazebo_px4_sim_ws.git
printf "/build/\n/devel/\n/.catkin_tools/\n*.pyc\n__pycache__/\n" > .gitignore
git add README.md src .gitignore
git commit -m "Add ArUco ROS workspace for PX4 landing demo"
git push -u origin main
```

## 5. 新设备快速部署

### 5.1 克隆主仓库和子模块

```bash
export PX4_DIR=$HOME/PX4-Visual-Landing
git clone --recursive git@github.com:zst1681/PX4-Visual-Landing.git "$PX4_DIR"
cd "$PX4_DIR"
git checkout visual-landing
git submodule update --init --recursive
```

### 5.2 安装依赖并构建

```bash
export PX4_DIR=$HOME/PX4-Visual-Landing
cd "$PX4_DIR"
bash Tools/setup/ubuntu.sh --no-nuttx
python3 -m pip install --user -r Tools/setup/requirements.txt
DONT_RUN=1 make px4_sitl_default gazebo
```

如果你有外部 ArUco 工作空间：

```bash
git clone git@github.com:zst1681/ros_gazebo_px4_sim_ws.git ~/ros_gazebo_px4_sim_ws-master
cd ~/ros_gazebo_px4_sim_ws-master
rosdep install --from-paths src --ignore-src -r -y
catkin build
```

### 5.3 启动视觉降落

```bash
export PX4_DIR=$HOME/PX4-Visual-Landing
cd "$PX4_DIR"
source scripts/setup_aruco_runtime.bash
roslaunch px4 aruco_search_and_land_demo.launch gui:=false
```

如果你的外部工作空间路径不是默认值：

```bash
cd "$PX4_DIR"
export ARUCO_WS=$HOME/workspaces/ros_gazebo_px4_sim_ws
export GAZEBO_WS=$HOME/catkin_ws
source scripts/setup_aruco_runtime.bash
roslaunch px4 aruco_search_and_land_demo.launch gui:=false
```

## 6. 验证命令

确认 ROS 能找到包：

```bash
source "$PX4_DIR/scripts/setup_aruco_runtime.bash"
rospack find px4
rospack find mavros
```

确认 MAVROS 连接：

```bash
rostopic echo -n 1 /mavros/state
```

确认图像和检测：

```bash
rostopic list | grep -E "camera|aruco|mavros/state|local_position"
rostopic hz /aruco/pose -w 5
```

跑一次 benchmark：

```bash
cd "$PX4_DIR"
source scripts/setup_aruco_runtime.bash
python3 scripts/benchmark_aruco_landing.py --runs 1 --timeout 140
```

清理残留进程：

```bash
scripts/cleanup_aruco_runtime.sh
```

## 7. 常见问题

`cv2.aruco` 不存在：安装 `libopencv-contrib-dev` 和 `python3-opencv`，并确认没有被 pip 里不带 contrib 的 `opencv-python` 覆盖。

`rospack find px4` 失败：先 `source scripts/setup_aruco_runtime.bash`，确认脚本没有报 `/opt/ros/noetic/setup.bash` 缺失。

Gazebo 找不到模型或世界：确认 `Tools/sitl_gazebo` 子模块已经指向你个人 fork 的提交，并执行过 `git submodule update --init --recursive`。

OFFBOARD/ARM 失败：确认 `/mavros/state` connected 为 `True`，当前 launch 里 MAVROS 已设置 `use_comp_id_system_control: true`，控制节点也会尝试设置 `COM_RCL_EXCEPT` 以允许无遥控 OFFBOARD。
