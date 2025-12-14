# VieNeu-TTS 后端服务

为 ASEAN-Workflow 提供越南语 TTS 接口的 FastAPI 后端服务。

## 快速开始

### 启动服务

直接用 Python 运行服务：

```bash
python services/tts_api.py
```

或指定自定义配置：

```bash
# 使用 GPU
TTS_BACKBONE_DEVICE=cuda \
TTS_CODEC_DEVICE=cuda \
python services/tts_api.py

# 使用 LMDeploy 优化（需要安装 lmdeploy）
TTS_ENABLE_LMDEPLOY=true \
python services/tts_api.py

# 自定义端口和输出目录
TTS_SERVICE_PORT=9000 \
TTS_OUTPUT_DIR=./audio_output \
python services/tts_api.py

# 添加 API Key 认证
TTS_API_KEY=your_secret_key \
python services/tts_api.py
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TTS_SERVICE_PORT` | 8000 | 服务端口 |
| `TTS_BACKBONE_REPO` | `pnnbao-ump/VieNeu-TTS` | 主干模型仓库 |
| `TTS_BACKBONE_DEVICE` | `cuda` / `cpu` | 主干设备（自动检测） |
| `TTS_CODEC_REPO` | `neuphonic/neucodec` | 编解码器仓库 |
| `TTS_CODEC_DEVICE` | `cuda` / `cpu` | 编解码器设备（自动检测） |
| `TTS_OUTPUT_DIR` | `outputs/tts_service` | 输出目录 |
| `TTS_API_KEY` | 无 | API Key（可选认证） |
| `TTS_ENABLE_LMDEPLOY` | `false` | 启用 LMDeploy 优化 |
| `TTS_ENABLE_TRITON` | `false` | 启用 Triton 编译加速 |

## API 接口

### 1. 合成语音

**请求**

```bash
POST /synthesize
```

```bash
curl -X POST http://localhost:8000/synthesize \
  -F "text=Xin chào, đây là bản demo" \
  -F "ref_text=Tuyên là một giọng nam miền Bắc" \
  -F "ref_codes_path=@./sample/Tuyên (nam miền Bắc).pt" \
  -F "emo_alpha=1.0"
```

**表单参数**

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `text` | string | ✅ | 要合成的文本 |
| `ref_text` | string | ✅ | 参考音频对应的文本 |
| `ref_codes` | string | ❌ | 预编码参考代码（Base64 或逗号分隔整数） |
| `ref_audio` | file | ❌ | 参考音频文件（如不提供 ref_codes 则必需） |
| `emo_alpha` | float | ❌ | 情感控制参数，默认 1.0 |
| `X-API-Key` | header | ❌ | API Key（如需认证） |

**响应**

```json
{
  "file_name": "output_a1b2c3d4.wav",
  "file_url": "http://localhost:8000/files/output_a1b2c3d4.wav"
}
```

### 2. 健康检查

```bash
GET /healthz
```

```bash
curl http://localhost:8000/healthz
```

响应：

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 3. 下载音频

```bash
GET /files/{file_name}
```

```bash
curl http://localhost:8000/files/output_a1b2c3d4.wav -o output.wav
```

## 从 ASEAN-Workflow 调用

### 方式 1：使用客户端库

```python
from ASEAN-Workflow.services.tts.vieneu_tts_client import synthesize_from_reference
from pathlib import Path

# 使用预编码的参考代码（推荐，更快）
output_wav = synthesize_from_reference(
    text="Xin chào, đây là bản demo",
    ref_text="Tuyên là một giọng nam miền Bắc",
    ref_codes_path=Path("./sample/Tuyên (nam miền Bắc).pt")
)

# 或使用参考音频（自动编码）
output_wav = synthesize_from_reference(
    text="Xin chào, đây là bản demo",
    ref_text="Tuyên là một giọng nam miền Bắc",
    ref_audio_path=Path("./sample/Tuyên (nam miền Bắc).wav")
)
```

### 方式 2：直接 HTTP 请求

```python
import requests

# 使用参考代码文件
with open("./sample/Tuyên (nam miền Bắc).pt", "rb") as f:
    ref_codes = f.read()

resp = requests.post(
    "http://localhost:8000/synthesize",
    data={
        "text": "Xin chào, đây là bản demo",
        "ref_text": "Tuyên là một giọng nam miền Bắc",
        "ref_codes": ref_codes,
        "emo_alpha": 1.0
    }
)

audio_url = resp.json()["file_url"]
print(f"Generated audio: {audio_url}")
```

### 方式 3：环境变量配置

在 ASEAN-Workflow 的 `.env` 中设置：

```env
VIENEU_SERVICE_URL=http://localhost:8000/synthesize
VIENEU_FILE_BASE=http://localhost:8000/files
VIENEU_DOWNLOAD_DIR=outputs/vieneu_tts_download
ENABLE_VIENEU_SERVICE=true
```

然后在代码中直接使用：

```python
from ASEAN-Workflow.services.tts.vieneu_tts_client import synthesize_from_reference

output_wav = synthesize_from_reference(
    text="你好",
    ref_text="参考文本",
    ref_codes_path=Path("./sample/voice.pt")
)
```

## 配置示例

### 最小配置（CPU）

```bash
python services/tts_api.py
```

### GPU 配置

```bash
TTS_BACKBONE_DEVICE=cuda \
TTS_CODEC_DEVICE=cuda \
python services/tts_api.py
```

### 生产环境配置（带 API Key）

```bash
TTS_SERVICE_PORT=9000 \
TTS_API_KEY=your_secret_key \
TTS_OUTPUT_DIR=/var/audio/tts_output \
TTS_BACKBONE_DEVICE=cuda \
TTS_CODEC_DEVICE=cuda \
python services/tts_api.py
```

### LMDeploy 优化配置（需额外安装）

```bash
# 先安装 lmdeploy
pip install lmdeploy

# 然后启动
TTS_ENABLE_LMDEPLOY=true \
TTS_ENABLE_TRITON=true \
python services/tts_api.py
```

## 文件结构

```
VieNeu-TTS/
├── services/
│   └── tts_api.py              # FastAPI 服务主文件
└── ASEAN-Workflow/
    └── services/
        └── tts/
            └── vieneu_tts_client.py  # ASEAN 侧调用客户端
```

## 常见问题

### Q: 如何使用自定义的模型？

A: 通过环境变量指定：

```bash
TTS_BACKBONE_REPO=path/to/your/model \
TTS_CODEC_REPO=path/to/your/codec \
python services/tts_api.py
```

### Q: 如何加速推理？

A: 几个选项：

1. 使用 GPU：`TTS_BACKBONE_DEVICE=cuda`
2. 使用 LMDeploy：`TTS_ENABLE_LMDEPLOY=true`
3. 使用 Triton：`TTS_ENABLE_TRITON=true`
4. 使用预编码的参考代码（比现场编码快）

### Q: 如何处理多个并发请求？

A: 目前服务使用单个 worker，可以通过以下方式改进：

```bash
# 使用 Gunicorn 多 worker（需先安装 gunicorn）
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:8000 services.tts_api:app
```

### Q: 生成的音频质量不好怎么办？

A: 

1. 检查参考文本是否准确
2. 检查参考音频文件质量
3. 尝试调整 `emo_alpha` 参数
4. 考虑对特定声音微调模型

## 依赖

- fastapi
- uvicorn
- torch
- transformers
- neucodec
- librosa
- soundfile
- requests

## 注意事项

- 首次启动会下载模型，可能需要较长时间
- GPU 推荐使用 NVIDIA 卡
- 生成的音频保存在 `TTS_OUTPUT_DIR`，定期清理以节省磁盘空间
- API Key 用于生产环境安全认证
