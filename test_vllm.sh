#!/bin/bash
################################################################################
# Quick vLLM Server Test Script
################################################################################

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

PORT=8000

echo -e "${BLUE}Testing vLLM Server on port $PORT...${NC}"
echo ""

# Test 1: List models
echo -e "${BLUE}[1/3] Testing /v1/models endpoint...${NC}"
MODELS=$(curl -s http://localhost:$PORT/v1/models 2>/dev/null)

if [ $? -eq 0 ] && echo "$MODELS" | grep -q "DeepSeek"; then
    MODEL_ID=$(echo "$MODELS" | python3 -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
    echo -e "${GREEN}✓ Models endpoint working${NC}"
    echo "  Model: $MODEL_ID"
else
    echo -e "${RED}✗ Models endpoint failed${NC}"
    exit 1
fi

echo ""

# Test 2: Simple completion
echo -e "${BLUE}[2/3] Testing text completion...${NC}"
RESPONSE=$(curl -s http://localhost:$PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/workspace/models/DeepSeek-R1-Distill-Qwen-14B",
        "prompt": "2+2=",
        "max_tokens": 10,
        "temperature": 0
    }' 2>/dev/null)

if [ $? -eq 0 ] && echo "$RESPONSE" | grep -q "text"; then
    TEXT=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null | head -1)
    echo -e "${GREEN}✓ Completion working${NC}"
    echo "  Response: 2+2=$TEXT"
else
    echo -e "${RED}✗ Completion failed${NC}"
    exit 1
fi

echo ""

# Test 3: Chat completion
echo -e "${BLUE}[3/3] Testing chat completion...${NC}"
CHAT=$(curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/workspace/models/DeepSeek-R1-Distill-Qwen-14B",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 5
    }' 2>/dev/null)

if [ $? -eq 0 ] && echo "$CHAT" | grep -q "content"; then
    CONTENT=$(echo "$CHAT" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
    echo -e "${GREEN}✓ Chat endpoint working${NC}"
    echo "  Response: $CONTENT"
else
    echo -e "${RED}✗ Chat failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=============================================================================="
echo -e "All tests passed! vLLM server is working correctly."
echo -e "=============================================================================="
echo -e "${NC}"

# Show GPU status
echo -e "${BLUE}GPU Status:${NC}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv | column -t -s ','
