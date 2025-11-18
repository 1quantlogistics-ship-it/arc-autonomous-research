# 🚀 DeepSeek-R1-Distill-Qwen-14B Quick Start

**Status:** ✅ Production Ready  
**Last Verified:** 2025-11-18  
**Hardware:** RunPod 2× NVIDIA A40 GPUs

---

## ⚡ Quick Commands

### Start the Server
```bash
cd /workspace/arc
./start_vllm.sh
```
⏱️ **Startup time:** ~6-8 minutes

### Test It's Working
```bash
./test_vllm.sh
```

### Stop the Server
```bash
./stop_vllm.sh
```

---

## 📚 Available Scripts

| Script | Purpose |
|--------|---------|
| **start_vllm.sh** | Complete startup with health checks |
| **stop_vllm.sh** | Graceful shutdown |
| **test_vllm.sh** | Run API tests |
| **VLLM_STARTUP_GUIDE.md** | Full documentation |
| **requirements_verified.txt** | Exact working package versions |

---

## 🧪 Quick Test

```bash
# Simple completion test
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/DeepSeek-R1-Distill-Qwen-14B",
    "prompt": "Explain AI in 10 words:",
    "max_tokens": 20
  }' | jq
```

---

## 📊 What's Running

- **Model:** DeepSeek-R1-Distill-Qwen-14B (14B parameters)
- **Server:** vLLM 0.3.3 (OpenAI-compatible API)
- **Port:** 8000
- **GPUs:** 2× A40 (tensor parallelism)
- **Memory:** ~30GB per GPU

---

## 🔍 Monitoring

```bash
# View logs
tail -f /workspace/arc/vllm.log

# Watch GPUs
watch -n 1 nvidia-smi

# Check status
curl http://localhost:8000/v1/models | jq
```

---

## ⚙️ Configuration

All settings in `start_vllm.sh` are **verified working**:

```bash
# Key parameters (DO NOT CHANGE unless needed)
TENSOR_PARALLEL_SIZE=2      # Use both GPUs
PORT=8000                    # API port
GPU_MEMORY_UTIL=0.9         # 90% GPU memory usage
MAX_SEQS=256                # Concurrent requests
```

---

## 🐛 Troubleshooting

### Server won't start
```bash
# Check logs
tail -100 /workspace/arc/vllm.log

# Kill any stuck processes
pkill -f vllm
./start_vllm.sh
```

### Port already in use
```bash
# Find what's using port 8000
ss -tlnp | grep 8000

# Stop it
./stop_vllm.sh
```

### Need more help?
See **VLLM_STARTUP_GUIDE.md** for detailed troubleshooting.

---

## 🔐 Security Note

The server binds to `0.0.0.0:8000` (all interfaces). For external access, use SSH tunnel:

```bash
# On your local machine
ssh -L 8000:localhost:8000 root@<runpod-ip> -p <port> -i ~/.ssh/id_ed25519

# Then access via
curl http://localhost:8000/v1/models
```

---

## 📦 Files Created

All scripts are in `/workspace/arc/`:

- ✅ `start_vllm.sh` - Startup script (executable)
- ✅ `stop_vllm.sh` - Stop script (executable)  
- ✅ `test_vllm.sh` - Test script (executable)
- ✅ `VLLM_STARTUP_GUIDE.md` - Full documentation
- ✅ `requirements_verified.txt` - Package versions
- ✅ `vllm.log` - Server logs (auto-created)
- ✅ `vllm.pid` - Process ID (auto-created)

---

## ✨ What Works

- ✅ vLLM server with tensor parallelism
- ✅ OpenAI-compatible API
- ✅ Text completions
- ✅ Chat completions
- ✅ Streaming responses
- ✅ Both GPUs utilized
- ✅ Auto health checks
- ✅ Graceful shutdown

## ⚠️ Known Issues

- ❌ ARC Control Plane (missing dependencies)
- ❌ ARC Dashboard (depends on Control Plane)

**But the core model serving is 100% operational!**

---

## 🎯 Next Steps

1. **Start the server:** `./start_vllm.sh`
2. **Test it works:** `./test_vllm.sh`
3. **Integrate with your app:** Use OpenAI SDK pointing to `localhost:8000`
4. **Monitor performance:** `watch nvidia-smi`

---

**Ready to use!** 🎉
