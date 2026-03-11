# 基于滤波器组切空间(Filter Bank Tangent Space)的实时运动想象脑机接口系统

## 项目简介

本项目实现了一个基于**滤波器组切空间(Filter Bank Tangent Space, FBTS)**算法的实时运动想象(Motor Imagery, MI)脑机接口(Brain-Computer Interface, BCI)系统。该系统通过黎曼几何方法处理脑电信号(EEG)，实现了高效、轻量级的四分类运动想象识别，可应用于实时游戏控制、辅助设备控制等场景。

### 核心特点

- **高效算法**：基于黎曼几何的切空间投影，无需深度学习的大量训练数据
- **实时性能**：单次推理延迟低于20ms，支持实时控制
- **轻量级**：可在普通CPU上运行，无需GPU加速
- **跨会话稳定性**：支持Session 1训练、Session 2测试的跨会话验证
- **完整应用**：包含数据采集、模型训练、实时控制、游戏应用全流程

---

## 技术原理

### 1. 滤波器组分解 (Filter Bank Decomposition)

脑电信号首先通过多个带通滤波器进行分解，提取不同频段的特征：
- 频段1: 8-10 Hz (mu节律)
- 频段2: 10-16 Hz (mu/beta过渡)
- 频段3: 16-24 Hz (beta节律)
- 频段4: 24-32 Hz (高频beta)

### 2. 协方差矩阵估计 (Covariance Estimation)

对每个频段的信号计算协方差矩阵，使用OAS(Oracle Approximating Shrinkage)估计器提高小样本下的估计稳定性。

### 3. 切空间投影 (Tangent Space Projection)

将协方差矩阵从黎曼流形投影到切空间，将非欧几里得数据转换为欧几里得特征向量：
- 参考点：使用几何均值作为参考点
- 投影：对数映射将 SPD 矩阵投影到切空间
- 向量化：提取上三角部分作为特征

### 4. 特征融合与分类 (Feature Fusion & Classification)

将各频段的切空间特征拼接，使用SVM进行分类：
- 特征选择：使用Fisher比率选择最具判别性的特征
- 分类器：支持向量机(SVM)配合RBF核

---

## 项目结构

```
MI_realtime_TangentSpace/
├── algorithms/                 # 算法实现
│   └── fbcsp.py               # FBCSP算法实现
├── config/                     # 配置文件
│   ├── mi_config.py           # 运动想象参数配置
│   └── algorithms_config.py   # 算法配置
├── experiments/                # 实验脚本
│   ├── ablation_study.py      # 消融实验
│   ├── latency_benchmark.py   # 延迟基准测试
│   └── statistical_analysis.py # 统计分析
├── data/                       # 数据存储
├── models/                     # 模型存储
├── results/                    # 实验结果
├── res/                        # 资源文件(图片等)
├── paper/                      # 论文相关文件
│
├── algorithms_collection.py    # 算法集合与模型定义
├── data_loader_moabb.py       # MOABB数据集加载
├── data_acquisition.py        # 实时数据采集
├── train_model.py             # 模型训练
├── calibrate_model.py         # 模型校准
├── evaluate_algorithms.py     # 算法评估
├── realtime_control.py        # 实时控制主程序
├── mi_maze_game.py            # 迷宫游戏
├── mi_tetris_game.py          # 俄罗斯方块游戏
└── mi_test_itr.py             # ITR测试
```

---

## 安装与配置

### 环境要求

- Python 3.8+
- 操作系统: Windows/Linux/macOS
- 硬件: 普通CPU即可，无需GPU

### 依赖安装

```bash
# 创建虚拟环境
conda create -n mi_bci python=3.10
conda activate mi_bci

# 安装核心依赖
pip install numpy scipy scikit-learn matplotlib
pip install mne pyriemann
pip install braindecode torch
pip install pygame pylsl
pip install moabb
pip install pandas seaborn
```

### OpenBCI配置

1. 使用OpenBCI Cyton板采集EEG信号
2. 8通道配置: C3, C4, Cz, F3, F4, Fz, T3, T4
3. 采样率: 250 Hz
4. 通过Lab Streaming Layer (LSL)传输数据

---

## 使用方法

### 1. 数据采集

运行数据采集程序，记录运动想象数据：

```bash
python data_acquisition.py --subject 1 --session 1
```

参数说明：
- `--subject`: 被试编号
- `--session`: 会话编号
- `--trials-per-class`: 每类试验次数(默认40)

操作流程：
1. 启动OpenBCI并确保LSL流正常运行
2. 运行数据采集脚本
3. 按空格键开始
4. 根据屏幕提示执行运动想象任务：
   - ↑ 舌头想象
   - ↓ 双脚想象
   - ← 左手想象
   - → 右手想象

### 2. 模型训练

使用采集的数据训练FBTS模型：

```bash
python train_model.py data/subject_1_session_1.mat --algorithm filterbank_tangent
```

参数说明：
- `mat_files`: 训练数据文件路径
- `--algorithm`: 算法选择 (`fbcsp` 或 `filterbank_tangent`)
- `--output-dir`: 模型输出目录

### 3. 模型校准

针对新会话数据进行快速校准：

```bash
python calibrate_model.py models/fbcsp_model data/subject_1_session_2.mat --eval
```

### 4. 实时控制

启动实时控制程序：

```bash
python realtime_control.py models/fbcsp_model
```

控制映射：
- 舌头想象 → 向上
- 双脚想象 → 向下
- 左手想象 → 向左
- 右手想象 → 向右

### 5. 游戏应用

#### 迷宫游戏

```bash
python mi_maze_game.py models/fbcsp_model
```

控制方式：
- 使用运动想象控制角色移动
- 舌头/双脚/左手/右手想象对应上下左右移动
- 找到出口即可获胜

#### 俄罗斯方块

```bash
python mi_tetris_game.py models/fbcsp_model
```

控制映射：
- 舌头想象 → 旋转方块
- 双脚想象 → 快速下落
- 左手想象 → 左移
- 右手想象 → 右移

### 6. ITR测试

测试信息传输率(Information Transfer Rate)：

```bash
python mi_test_itr.py models/fbcsp_model --trials 20
```

---

## 核心代码说明

### algorithms_collection.py

包含所有算法的实现：

- `FilterBankTangentSpace`: 滤波器组切空间算法（核心算法）
- `get_algorithm()`: 获取指定算法的实例
- 支持算法：CSP+LDA, FBCSP, MDM, RiemannTangentSpace, EEGNet, EEGTCNet等

### realtime_control.py

实时控制主程序：

- `connect_lsl()`: 连接LSL数据流
- `load_model()`: 加载训练好的模型
- `classify_window()`: 对滑动窗口数据进行分类
- `main_loop()`: 主控制循环

### data_acquisition.py

数据采集模块：

- `collect_mi_data()`: 采集运动想象数据
- `run_trial()`: 运行单次试验
- `save_mat()`: 保存数据为MAT格式

### evaluate_algorithms.py

算法评估脚本：

- 支持多种算法的交叉验证
- 生成混淆矩阵、t-SNE可视化
- 输出准确率、Kappa系数等指标

---

## 算法对比

| 算法 | 准确率(BCI IV 2A) | 训练时间 | 推理时间 | GPU需求 |
|------|------------------|----------|----------|---------|
| FBTS+SVM | 72.85% | 5.59s | 17.04ms | 否 |
| EEGNet | 58.48% | ~10min | ~50ms | 是 |
| EEGTCNet | 59.88% | ~15min | ~60ms | 是 |
| EEGITNet | 44.98% | ~12min | ~55ms | 是 |
| IFNet | 67.09% | ~20min | ~70ms | 是 |
| FBCSP | 68.22% | 8.23s | 15.23ms | 否 |

*注：深度学习算法在跨会话场景下性能下降明显，而FBTS保持稳定*

---

## 性能指标

### BCI IV 2a数据集

- **准确率**: 72.85% (跨会话: Session 1训练, Session 2测试)
- **Kappa系数**: 0.638
- **训练时间**: 5.59秒（含超参数优化）
- **推理时间**: 17.04毫秒/次

### PhysioNet MI数据集

- **准确率**: 51.95% (109名被试，四分类随机概率25%)
- **训练时间**: 6.47秒

### 实时性能

- **在线准确率**: 90%
- **ITR**: 20.59 bits/min
- **延迟**: <20ms

---

## 实验脚本

### 消融实验

```bash
python experiments/ablation_study.py
```

分析不同组件对性能的影响。

### 延迟基准测试

```bash
python experiments/latency_benchmark.py
```

测试系统端到端延迟。

### 统计分析

```bash
python experiments/statistical_analysis.py
```

生成统计报告和显著性检验。

---

## 注意事项

1. **数据质量**：确保电极阻抗低于5kΩ，以获得良好的信号质量
2. **被试训练**：新被试需要一定的训练时间来掌握运动想象技巧
3. **环境控制**：避免电磁干扰，保持安静的环境
4. **模型保存**：训练好的模型保存在`models/`目录下
5. **结果查看**：实验结果保存在`results/`目录下

---

## 引用

如果您使用了本项目，请引用：

```bibtex
@article{fbts2024,
  title={Real-Time Motor Imagery BCI Using Filter Bank Tangent Space},
  author={Your Name},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  year={2024}
}
```

---

## 许可证

本项目采用 MIT 许可证。

---

## 联系方式

如有问题或建议，请通过GitHub Issues联系。

---

## 致谢

- [MOABB](https://github.com/NeuroTechX/moabb): 脑机接口算法基准测试平台
- [PyRiemann](https://github.com/pyRiemann/pyRiemann): 黎曼几何工具包
- [MNE-Python](https://mne.tools/): 脑电信号处理工具包
- [Braindecode](https://braindecode.ai/): 深度学习脑电解码库
