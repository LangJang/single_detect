# YOLO 目标检测系统

基于 YOLOv7 的实时目标检测系统，提供 PySide6 桌面应用，支持单视频、批量、监控三种检测模式，附带场景管理、ROI 过滤、邮件告警等功能。

## 功能

- **三种检测模式**：单视频检测、批量处理、目录监控（自动检测新文件）
- **场景 (Tag) 管理**：保存/加载检测参数组合（模型、置信度、ROI、标定、告警阈值），一键切换
- **ROI 区域过滤**：多边形 ROI，支持质心/重叠/完全包含三种判定策略
- **相机标定**：像素坐标 → 世界坐标转换，估算目标实际尺寸
- **邮件告警**：检测到异常时通过 SMTP 发送邮件通知
- **SQLite 持久化**：场景配置和处理记录存储在 SQLite，支持 WAL 模式并发读写
- **PySide6 桌面应用**：原生 Qt 界面，支持图片点击选取 ROI、视频预览

## 环境要求

- Python 3.10+
- PyTorch 1.12+
- CUDA（可选，CPU 亦可运行）

## 安装

```bash
git clone https://github.com/LangJang/single_detect.git
cd single_detect
pip install -r requirements.txt
```

## 模型准备

将 YOLOv7 权重文件 (`.pth`) 放入 `models/` 目录，默认使用 `ep950-loss0.050-val_loss0.055.pth`。

```
models/
├── ep950-loss0.050-val_loss0.055.pth
└── ep400-loss0.049-val_loss0.034.pth
```

## 启动

```bash
python main.py
```

启动后自动打开桌面窗口。

## 项目结构

```
single_detect/
├── main.py                  # 入口，启动 Qt 应用
├── detector.py              # YOLOv7 检测器封装
├── video_processor.py       # 视频处理、ROI 过滤、告警评估
├── scene_config.py          # 场景配置数据类
├── scene_manager.py         # 场景 (Tag) 增删改查 + 模型切换
├── shared_state.py          # 全局共享状态
├── database.py              # SQLite 持久化层
├── database_migration.py    # JSON → SQLite 迁移
├── file_scanner.py          # 视频文件扫描（时间范围过滤）
├── calibration.py           # 相机标定（像素 ↔ 世界坐标）
├── geometry.py              # 坐标解析与绘制工具
├── alert_mail.py            # SMTP 邮件告警
├── ui/
│   ├── main_window.py       # Qt 主窗口（控件 + 布局 + 信号槽）
│   ├── workers.py           # QThread 后台任务（监控/单视频/批量）
│   └── image_label.py       # 可点击图片标签（ROI 选点 + 多边形绘制）
├── handlers/
│   ├── single.py            # 单视频检测逻辑
│   ├── batch.py             # 批量检测逻辑
│   ├── monitor.py           # 目录监控逻辑
│   └── preview.py           # 视频预览 + ROI 选取
├── nets/
│   ├── yolo.py              # YOLOv7 网络定义
│   └── backbone.py          # 骨干网络
├── utils/
│   ├── utils.py             # 图像预处理工具
│   └── utils_bbox.py        # 边界框解码 & NMS
├── model_data/
│   ├── coco_classes.txt     # COCO 类别名称
│   ├── yolo_anchors.txt     # Anchor 配置
│   └── simhei.ttf           # 中文字体
└── models/                  # 模型权重文件（需自行放入）
```

## 架构

```
main.py  →  QApplication
              └── MainWindow (ui/main_window.py)
                    ├── 场景管理 ─── scene_manager.py ─── database.py
                    ├── ROI 预览 ─── handlers/preview.py ─── geometry.py
                    ├── 检测参数 ─── detector.py / video_processor.py
                    └── 三个处理 Tab
                         ├── 监控  ─── MonitorWorker (QThread)
                         ├── 单视频 ─── SingleVideoWorker (QThread)
                         └── 批量  ─── BatchWorker (QThread)
```

## License

MIT
