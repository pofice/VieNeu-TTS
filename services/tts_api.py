"""
VieNeu-TTS FastAPI Service
提供 HTTP 接口供 ASEAN-Workflow 调用越南语 TTS
"""

import os
import base64
import tempfile
import uuid
from pathlib import Path
from typing import Optional
import io

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# 导入 VieNeuTTS（不修改现有代码，直接使用）
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vieneu_tts import VieNeuTTS, FastVieNeuTTS

# ============================================================================
# CONFIGURATION
# ============================================================================

SERVICE_PORT = int(os.getenv("TTS_SERVICE_PORT", "8000"))
BACKBONE_REPO = os.getenv("TTS_BACKBONE_REPO", "pnnbao-ump/VieNeu-TTS")
BACKBONE_DEVICE = os.getenv("TTS_BACKBONE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
CODEC_REPO = os.getenv("TTS_CODEC_REPO", "neuphonic/neucodec")
CODEC_DEVICE = os.getenv("TTS_CODEC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = Path(os.getenv("TTS_OUTPUT_DIR", "outputs/tts_service")).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("TTS_API_KEY")
ENABLE_LMDEPLOY = os.getenv("TTS_ENABLE_LMDEPLOY", "false").lower() == "true"
ENABLE_TRITON = os.getenv("TTS_ENABLE_TRITON", "false").lower() == "true"

# ============================================================================
# GLOBAL STATE
# ============================================================================

tts_model = None

# ============================================================================
# UTILITIES
# ============================================================================

def load_tts_model():
    """加载 TTS 模型"""
    global tts_model
    
    if tts_model is not None:
        return tts_model
    
    print(f"🚀 Loading VieNeu-TTS Model...")
    print(f"   Backbone: {BACKBONE_REPO} on {BACKBONE_DEVICE}")
    print(f"   Codec: {CODEC_REPO} on {CODEC_DEVICE}")
    print(f"   LMDeploy: {ENABLE_LMDEPLOY}")
    
    try:
        if ENABLE_LMDEPLOY:
            tts_model = FastVieNeuTTS(
                backbone_repo=BACKBONE_REPO,
                backbone_device=BACKBONE_DEVICE,
                codec_repo=CODEC_REPO,
                codec_device=CODEC_DEVICE,
                enable_triton=ENABLE_TRITON,
            )
            print("   ✅ FastVieNeuTTS (LMDeploy) loaded")
        else:
            tts_model = VieNeuTTS(
                backbone_repo=BACKBONE_REPO,
                backbone_device=BACKBONE_DEVICE,
                codec_repo=CODEC_REPO,
                codec_device=CODEC_DEVICE,
            )
            print("   ✅ VieNeuTTS (Transformers) loaded")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        raise
    
    return tts_model

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """验证 API Key"""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

def decode_ref_codes(codes_data: str) -> np.ndarray | torch.Tensor:
    """
    从字符串解码 ref_codes
    支持格式：
    - Base64 编码的 .pt 文件
    - 逗号分隔的整数列表
    """
    try:
        # 尝试作为 Base64 解码（来自二进制 .pt 文件）
        decoded_bytes = base64.b64decode(codes_data)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
            f.write(decoded_bytes)
            f.flush()
            temp_path = f.name
        
        ref_codes = torch.load(temp_path, map_location="cpu")
        os.unlink(temp_path)
        return ref_codes
    except Exception:
        pass
    
    try:
        # 尝试作为逗号分隔的整数列表
        codes_list = [int(x.strip()) for x in codes_data.split(",")]
        return np.array(codes_list, dtype=np.int32)
    except Exception as e:
        raise ValueError(f"Failed to decode ref_codes: {e}")

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="VieNeu-TTS Service",
    description="越南语文本转语音服务 (TTS Service for Vietnamese)",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    load_tts_model()

@app.get("/healthz")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "model_loaded": tts_model is not None}

@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    ref_text: str = Form(...),
    ref_codes: Optional[str] = Form(None),
    ref_audio: Optional[UploadFile] = File(None),
    emo_alpha: float = Form(1.0),
    x_api_key: Optional[str] = Header(None),
):
    """
    合成语音接口
    
    参数:
        text: 要合成的文本 (必需)
        ref_text: 参考音频对应的文本 (必需)
        ref_codes: 预编码的参考代码，Base64 或逗号分隔的整数 (可选)
        ref_audio: 参考音频文件 (可选，如果不提供 ref_codes 则必需)
        emo_alpha: 情感控制参数 (默认 1.0)
        x_api_key: API Key 用于认证 (可选)
    
    返回:
        {
            "file_name": "output_xxxxx.wav",
            "file_url": "http://localhost:8000/files/output_xxxxx.wav"
        }
    """
    # 验证 API Key
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    
    # 验证输入
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text is required and cannot be empty")
    if not ref_text or not ref_text.strip():
        raise HTTPException(status_code=400, detail="ref_text is required and cannot be empty")
    
    try:
        tts = tts_model or load_tts_model()
        
        # 处理 ref_codes
        if ref_codes:
            # 使用提供的 ref_codes
            print(f"Using provided ref_codes")
            ref_codes_processed = decode_ref_codes(ref_codes)
        elif ref_audio:
            # 从上传的音频文件编码
            print(f"Encoding reference audio from upload...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                audio_data = await ref_audio.read()
                f.write(audio_data)
                f.flush()
                temp_audio_path = f.name
            
            try:
                ref_codes_processed = tts.encode_reference(temp_audio_path)
            finally:
                os.unlink(temp_audio_path)
        else:
            raise HTTPException(
                status_code=400,
                detail="Either ref_codes or ref_audio must be provided"
            )
        
        # 执行推理
        print(f"Synthesizing: {text[:50]}...")
        wav = tts.infer(
            text=text,
            ref_codes=ref_codes_processed,
            ref_text=ref_text
        )
        
        # 保存输出
        file_name = f"output_{uuid.uuid4().hex[:8]}.wav"
        output_path = OUTPUT_DIR / file_name
        sf.write(str(output_path), wav, 24000)
        
        print(f"✅ Generated: {file_name}")
        
        return {
            "file_name": file_name,
            "file_url": f"http://localhost:{SERVICE_PORT}/files/{file_name}"
        }
    
    except Exception as e:
        print(f"❌ Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{file_name}")
async def download_file(file_name: str):
    """下载生成的音频文件"""
    file_path = OUTPUT_DIR / file_name
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=file_name,
        media_type="audio/wav"
    )

# 挂载静态文件目录（用于直接访问）
try:
    app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")
except Exception:
    pass

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print(f"""
╔════════════════════════════════════════════════════════════╗
║          VieNeu-TTS FastAPI Service                        ║
║  越南语文本转语音服务 (Vietnamese TTS Service)              ║
╚════════════════════════════════════════════════════════════╝

Configuration:
  - Port: {SERVICE_PORT}
  - Backbone: {BACKBONE_REPO}
  - Backbone Device: {BACKBONE_DEVICE}
  - Codec: {CODEC_REPO}
  - Codec Device: {CODEC_DEVICE}
  - Output Directory: {OUTPUT_DIR}
  - API Key Required: {bool(API_KEY)}
  - LMDeploy: {ENABLE_LMDEPLOY}
  - Triton: {ENABLE_TRITON}

Endpoints:
  - POST /synthesize
  - GET /healthz
  - GET /files/{{file_name}}

Starting server...
""")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=SERVICE_PORT,
        workers=1,
        timeout_keep_alive=600
    )
