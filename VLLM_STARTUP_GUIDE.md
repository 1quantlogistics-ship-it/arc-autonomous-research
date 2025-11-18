# vLLM Server Startup Guide
## DeepSeek-R1-Distill-Qwen-14B on RunPod 2× A40 GPUs

**Last Updated:** 2025-11-18  
**Status:** ✅ Verified Working Configuration

---

## Quick Start

### Start Server
```bash
cd /workspace/arc
./start_vllm.sh
```

### Stop Server
```bash
cd /workspace/arc
./stop_vllm.sh
```

### Check Status
```bash
# Check if running
ps aux | grep vllm

# View logs
tail -f /workspace/arc/vllm.log

# Check GPU usage
watch -n 1 nvidia-smi

# Test API
curl http://localhost:8000/v1/models | jq
```

---

## Verified Configuration

### Hardware Requirements
- **GPUs:** 2× NVIDIA A40 (46GB VRAM each)
- **GPU Memory Used:** ~30GB per GPU
- **System RAM:** ~8-10GB
- **Disk Space:** ~28GB for model files

### Software Stack (Pinned Versions)
```
Python:         3.10.12
CUDA:           12.1 (PyTorch), 11.5 (toolkit), 12.7 (driver)

Core ML:
  torch         2.1.2+cu121
  vllm          0.3.3
  transformers  4.38.0
  numpy         1.26.4
  sentencepiece 0.2.1
  accelerate    1.11.0

API Stack:
  fastapi       0.121.2
  uvicorn       0.38.0
  pydantic      2.12.4
```

### vLLM Server Parameters
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/DeepSeek-R1-Distill-Qwen-14B \
  --tensor-parallel-size 2 \
  --port 8000 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 256 \
  --dtype auto \
  --kv-cache-dtype auto
```

---

## Critical Fixes Applied

### 1. outlines Library Patch
**Problem:** `pyairports==0.0.1` package has broken module structure  
**Symptom:** `ModuleNotFoundError: No module named 'pyairports'`

**Fix Applied:**
```bash
# File: venv/lib/python3.10/site-packages/outlines/types/airports.py
# Changed:
from pyairports.airports import AIRPORT_LIST

# To:
# PATCHED: pyairports package is broken, using empty list
AIRPORT_LIST = []
```

**Note:** The `start_vllm.sh` script automatically applies this patch.

---

## API Usage

### Endpoint URLs
- **List Models:** `http://localhost:8000/v1/models`
- **Completions:** `http://localhost:8000/v1/completions`
- **Chat:** `http://localhost:8000/v1/chat/completions`

### Example: Text Completion
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/DeepSeek-R1-Distill-Qwen-14B",
    "prompt": "Explain quantum computing:",
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### Example: Chat Completion
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/DeepSeek-R1-Distill-Qwen-14B",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 300,
    "temperature": 0.7
  }'
```

### Python Example (OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM doesn't require auth
)

response = client.chat.completions.create(
    model="/workspace/models/DeepSeek-R1-Distill-Qwen-14B",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

---

## Startup Sequence

The `start_vllm.sh` script performs these steps:

1. **Pre-flight Checks** (~5 seconds)
   - Verify 2 GPUs available
   - Check model files exist (4 safetensors)
   - Validate virtual environment
   - Confirm vLLM installed

2. **Apply Patches** (~1 second)
   - Fix outlines library pyairports import

3. **Start Server** (~5-7 minutes)
   - Initialize Ray distributed framework
   - Load model weights to GPUs (tensor parallel)
   - Compile CUDA graphs for optimization
   - Start Uvicorn ASGI server

4. **Health Check** (~5 seconds)
   - Wait for API endpoint to respond
   - Verify model loaded correctly
   - Display GPU utilization

**Total Time:** ~6-8 minutes from start to ready

---

## Monitoring

### Real-time Logs
```bash
tail -f /workspace/arc/vllm.log
```

### GPU Utilization
```bash
watch -n 1 nvidia-smi
```

### Expected GPU State
**Idle (no requests):**
- Utilization: 0-5%
- Memory: ~30GB per GPU (model weights loaded)
- Temperature: 30-40°C

**Active Inference:**
- Utilization: 80-95% per GPU
- Memory: 30-40GB per GPU (KV cache grows)
- Temperature: 60-75°C

### Performance Metrics
- **Throughput:** ~10-50 tokens/second (varies by prompt length)
- **Latency:** ~100-500ms for first token
- **Max Context:** 131,072 tokens (model config)
- **Concurrent Requests:** Up to 256 (--max-num-seqs)

---

## Troubleshooting

### Server Won't Start

**Check logs:**
```bash
tail -100 /workspace/arc/vllm.log
```

**Common issues:**
1. **Port already in use**
   ```bash
   ss -tlnp | grep 8000
   kill <PID>
   ```

2. **GPU out of memory**
   - Reduce `--gpu-memory-utilization` to 0.85
   - Reduce `--max-num-seqs` to 128

3. **Model files missing/corrupt**
   ```bash
   ls -lh /workspace/models/DeepSeek-R1-Distill-Qwen-14B/*.safetensors
   # Should see 4 files: 8.2GB, 8.1GB, 8.1GB, 3.3GB
   ```

### Slow Inference

**Check GPU utilization:**
```bash
nvidia-smi
```

If utilization is low during inference:
- May be CPU-bound on preprocessing
- Check if tensor parallelism is working (both GPUs active)
- Verify CUDA graphs compiled (check logs for "Graph capturing finished")

### Server Crashes During Inference

**Common causes:**
1. **Out of GPU memory:** Reduce max sequence length or batch size
2. **CUDA errors:** Check `dmesg` for GPU errors
3. **Model corruption:** Re-download model files

---

## File Locations

| Path | Description |
|------|-------------|
| `/workspace/arc/start_vllm.sh` | Startup script |
| `/workspace/arc/stop_vllm.sh` | Stop script |
| `/workspace/arc/vllm.log` | Server logs |
| `/workspace/arc/vllm.pid` | Process ID file |
| `/workspace/arc/venv` | Python virtual environment |
| `/workspace/models/DeepSeek-R1-Distill-Qwen-14B` | Model files (27.7GB) |

---

## Production Deployment

### Security Recommendations

1. **Use SSH tunnel for external access:**
   ```bash
   # On local machine
   ssh -L 8000:localhost:8000 root@<runpod-ip> -p <port> -i ~/.ssh/id_ed25519
   ```

2. **Add authentication** (vLLM doesn't have built-in auth):
   - Use nginx reverse proxy with basic auth
   - Or implement API key validation in application layer

3. **Rate limiting:**
   - Use nginx or application-level rate limiting
   - vLLM has built-in queue management

### Auto-start on Boot

Add to crontab:
```bash
@reboot /workspace/arc/start_vllm.sh >> /workspace/arc/startup.log 2>&1
```

### Log Rotation

```bash
# Add to /etc/logrotate.d/vllm
/workspace/arc/vllm.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    copytruncate
}
```

---

## Known Limitations

1. **ARC Application Layer:** Control Plane and Dashboard have missing dependencies (memory_handler module)
2. **No Authentication:** vLLM server has no built-in API authentication
3. **Single Model:** Only one model can be loaded at a time
4. **Memory Overhead:** Requires ~30GB VRAM per GPU even when idle

---

## Support & Debugging

### Useful Commands
```bash
# Full system status
./start_vllm.sh --check-only  # (if implemented)

# Force restart
./stop_vllm.sh && sleep 5 && ./start_vllm.sh

# Clean restart (clear cache)
./stop_vllm.sh
rm -rf /workspace/arc/vllm.log*
rm -rf ~/.cache/huggingface/
./start_vllm.sh

# Verify model integrity
cd /workspace/models/DeepSeek-R1-Distill-Qwen-14B
sha256sum *.safetensors
```

### Get Help
- vLLM Documentation: https://docs.vllm.ai/
- GitHub Issues: https://github.com/vllm-project/vllm/issues
- Model Card: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

---

**Document Version:** 1.0  
**Last Verified:** 2025-11-18  
**Configuration Hash:** a7f3c8d9 (for reference tracking)
