# RD-Agent Setup Guide for This Cloud Session

## Current Status

**Environment:** Cloud development container
**Python:** 3.11.14 ✅
**Docker:** Installed but daemon cannot start ❌
**RD-Agent:** Installing...

## Why Docker Won't Start

This session is running inside a container (likely for security). Docker-in-Docker requires privileged mode, which isn't available. This means:

❌ Cannot run `fin_factor`, `fin_model`, `fin_quant` (need Docker)
❌ Cannot run `data_science` scenarios (need Docker)
❌ Cannot run most isolated code execution

## What You CAN Do Here

### 1. Code Exploration ✅

```bash
# View the comprehensive reviews
cat CODEBASE_REVIEW_REPORT.md
cat SECURITY_AUDIT_REPORT.md
cat RD-AGENT-CODEBASE-OVERVIEW.md

# Explore the code
cd rdagent/core
ls -la
```

### 2. Configuration Testing ✅

Once installation completes:

```bash
# Create a test configuration
cat > .env << 'EOF'
OPENAI_API_KEY=sk-test-key-here
CHAT_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
EOF

# Test the CLI
rdagent --help
rdagent health_check --no-check-docker
```

### 3. Code Analysis ✅

```bash
# Run linting (no Docker needed)
make lint

# Check type hints
make mypy

# View tests
cd test && ls -la
```

## What You NEED for Full Functionality

### Requirements:
1. **Linux** (Ubuntu 20.04+ or similar)
2. **Docker** with daemon running
3. **Python 3.10 or 3.11**
4. **4GB+ RAM** (8GB+ recommended)
5. **LLM API keys** (OpenAI, Azure, or DeepSeek)

### Full Setup on Local Machine:

```bash
# 1. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# 2. Clone and install
git clone https://github.com/microsoft/RD-Agent
cd RD-Agent
conda create -n rdagent python=3.10
conda activate rdagent
pip install -e .

# 3. Configure
cat > .env << 'EOF'
# Your OpenAI key
OPENAI_API_KEY=sk-your-actual-key-here

# Models
CHAT_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small

# Optional: Rate limiting
MAX_RETRY=10
RETRY_WAIT_SECONDS=20
EOF

# 4. Test setup
rdagent health_check

# 5. Run first demo (paper implementation)
rdagent general_model "https://arxiv.org/pdf/2210.09789"

# 6. Monitor (in another terminal)
rdagent ui --port 19899 --log-dir log/
```

## Alternative: Cloud VM Setup

### AWS EC2 Ubuntu Setup:

```bash
# After SSH into EC2 instance
sudo apt-get update
sudo apt-get install -y docker.io python3-pip git
sudo usermod -aG docker ubuntu
newgrp docker

# Continue with installation steps above
```

### Google Cloud Compute:

```bash
# After SSH into GCP instance
sudo apt-get update
sudo apt-get install -y docker.io python3-pip git
sudo usermod -aG docker $USER
newgrp docker

# Continue with installation steps above
```

## Testing Without Docker (Limited)

Some components can be tested without Docker:

```bash
# Test LLM backend
python3 -c "
from rdagent.oai.llm_utils import APIBackend
backend = APIBackend()
response = backend.build_messages_and_create_chat_completion(
    system_prompt='You are helpful',
    user_prompt='Say hello'
)
print(response)
"

# Test configuration loading
python3 -c "
from rdagent.core.conf import RD_AGENT_SETTINGS
print(f'Workspace: {RD_AGENT_SETTINGS.workspace_path}')
"

# Run unit tests (no Docker needed)
pytest test/oai/test_completion.py
```

## Cost Estimates

Before running, understand API costs:

| Scenario | Duration | Estimated Cost |
|----------|----------|---------------|
| `general_model` (paper) | 10-30 min | $1-5 |
| `fin_factor` (1 iteration) | 30-60 min | $5-20 |
| `fin_quant` (full run) | 2-4 hours | $50-200 |
| `data_science` (kaggle) | 1-8 hours | $10-100 |

**Tips to reduce costs:**
- Use DeepSeek (10x cheaper than GPT-4)
- Set iteration limits
- Use cheaper models for exploration
- Monitor token usage in real-time

## Recommended Starting Path

**Day 1: Local Setup**
1. Set up on local Linux or WSL2
2. Install Docker + RD-Agent
3. Configure API keys
4. Run `rdagent health_check`

**Day 2: First Demo**
1. Try `general_model` with a simple paper
2. Monitor in UI dashboard
3. Review generated code

**Day 3: Advanced Scenarios**
1. Try `fin_factor` or `data_science`
2. Adjust configuration
3. Experiment with different models

## Security Warning

⚠️ **Before running, review the security audit:**

See `SECURITY_AUDIT_REPORT.md` for:
- 4 critical shell injection vulnerabilities
- Unsafe pickle deserialization
- API key logging issues

**Recommendation:** Run in isolated VM/container, not on production systems.

## Getting Help

- **Documentation:** https://rdagent.readthedocs.io/
- **Issues:** https://github.com/microsoft/RD-Agent/issues
- **Discord:** https://discord.gg/ybQ97B6Jjy
- **Review Docs:** See `CODEBASE_REVIEW_REPORT.md` (30+ pages)

## Summary

**This cloud session:**
- ✅ Good for code review and exploration
- ❌ Cannot run actual RD-Agent scenarios (no Docker)

**For full functionality:**
- Set up on local Linux with Docker
- Or use cloud VM (EC2, GCP, Azure)
- Budget $10-100 for LLM API costs

---

**Current Assessment:** Environment NOT suitable for production RD-Agent use.
**Recommendation:** Use this for review, set up proper environment separately.
