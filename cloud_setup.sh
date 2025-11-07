#!/bin/bash
# Cloud Session Setup for RD-Agent
# Configures RD-Agent to run in Anthropic cloud session without Docker

set -e

# Get session ID from argument or environment
SESSION_ID="${1:-${CLOUD_SESSION_ID:-default}}"

echo "🌩️  Setting up RD-Agent for Cloud Session"
echo "=========================================="
echo "Session ID: $SESSION_ID"

# Detect if we're in a cloud/container environment
if [ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
    echo "✓ Cloud/container environment detected"
fi

# Check Docker availability
if ! docker ps >/dev/null 2>&1; then
    echo "⚠️  Docker daemon not available (expected in cloud session)"
    echo "   → Will use LocalEnv instead"
fi

# Create .env for cloud session
echo ""
echo "📝 Creating cloud-optimized .env configuration..."

cat > .env << 'ENVEOF'
# Cloud Session Configuration for RD-Agent
# ==========================================

# LLM Configuration
# -----------------
# Option 1: Use your real API key for full functionality
# OPENAI_API_KEY=sk-your-actual-key-here
# CHAT_MODEL=gpt-4o
# EMBEDDING_MODEL=text-embedding-3-small

# Option 2: Use test mode (limited functionality, no real LLM calls)
OPENAI_API_KEY=test-key-no-real-calls
CHAT_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small

# Force LocalEnv (no Docker)
# ---------------------------
FORCE_LOCAL_ENV=true
USE_DOCKER=false

# Qlib Configuration for LocalEnv
# --------------------------------
QLIB_FACTOR_ENV_TYPE=local
QLIB_MODEL_ENV_TYPE=local

# Rate Limiting (for real API usage)
# -----------------------------------
MAX_RETRY=10
RETRY_WAIT_SECONDS=20

# Cloud Session
# -------------
CLOUD_SESSION_ID=$SESSION_ID

# Logging
# -------
LOG_LEVEL=INFO
LOG_PATH=./log/$SESSION_ID

# Workspace
# ---------
WORKSPACE_PATH=./git_ignore_folder/$SESSION_ID

# Cache (optional)
# ----------------
USE_CHAT_CACHE=False
USE_EMBEDDING_CACHE=False
ENVEOF

echo "✓ Created .env configuration"

# Check Python dependencies
echo ""
echo "🐍 Checking Python dependencies..."

MISSING_DEPS=()

python3 -c "import pandas" 2>/dev/null || MISSING_DEPS+=("pandas")
python3 -c "import numpy" 2>/dev/null || MISSING_DEPS+=("numpy")
python3 -c "import sklearn" 2>/dev/null || MISSING_DEPS+=("scikit-learn")
python3 -c "import dill" 2>/dev/null || MISSING_DEPS+=("dill")

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "⚠️  Missing dependencies: ${MISSING_DEPS[*]}"
    echo "   Installing..."
    pip install -q ${MISSING_DEPS[*]}
    echo "✓ Dependencies installed"
else
    echo "✓ All dependencies present"
fi

# Create session-specific directories
mkdir -p "log/$SESSION_ID" "git_ignore_folder/$SESSION_ID"
echo "✓ Created session workspace: log/$SESSION_ID, git_ignore_folder/$SESSION_ID"

# Create cloud demo runner
echo ""
echo "🚀 Creating cloud demo runner..."

cat > run_cloud_demo.py << 'PYEOF'
#!/usr/bin/env python3
"""
Cloud Session Demo Runner for RD-Agent
Runs a simplified R&D loop that works in cloud environments without Docker

Usage:
    python3 run_cloud_demo.py [session_id]

Example:
    python3 run_cloud_demo.py session_011CUuCFvEbn4PTLEwJcWkX9
"""

import os
import sys
from pathlib import Path

# Get session ID from args or environment
SESSION_ID = sys.argv[1] if len(sys.argv) > 1 else os.getenv('CLOUD_SESSION_ID', 'default')

# Ensure .env is loaded
from dotenv import load_dotenv
load_dotenv()

def check_environment():
    """Check if environment is properly configured"""
    print("🔍 Environment Check")
    print("=" * 60)

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key == "test-key-no-real-calls":
        print("⚠️  Using TEST mode (no real LLM calls)")
        print("   To use real LLM: export OPENAI_API_KEY=sk-your-key")
        return False
    elif api_key.startswith("sk-"):
        print("✓ Real API key detected")
        return True
    else:
        print("❌ No valid API key found")
        print("   Set: export OPENAI_API_KEY=sk-your-key")
        return False

def run_simple_demo():
    """Run a simple LocalEnv demo"""
    print("\n🎯 Running Simple LocalEnv Demo")
    print("=" * 60)

    from rdagent.utils.env import LocalConf, LocalEnv
    from pathlib import Path
    import tempfile

    # Create temp workspace
    workspace = Path(tempfile.mkdtemp(prefix="rdagent_cloud_"))

    # Create a simple test script
    test_script = workspace / "test.py"
    test_script.write_text("""
import sys
import os
print("✓ RD-Agent LocalEnv working in cloud session!")
print(f"  Python: {sys.version.split()[0]}")
print(f"  Working directory: {os.getcwd()}")
""")

    # Configure and run
    config = LocalConf(default_entry=sys.executable)
    env = LocalEnv(conf=config)

    result = env.run(
        entry=f"{sys.executable} test.py",
        local_path=str(workspace)
    )

    print(result.stdout if hasattr(result, 'stdout') else str(result))
    print("\n✓ LocalEnv demo complete")

def run_factor_demo():
    """Run a simple factor generation demo (mock)"""
    print("\n🔬 Factor Research Demo (Simplified)")
    print("=" * 60)

    # Show what RD-Agent would do with real LLM
    print("""
    Phase 1: Hypothesis Generation
    ─────────────────────────────────
    📊 Analyzing 2025 market conditions...
    💡 Generated hypothesis: "Volatility-adaptive momentum factor"

    Phase 2: Experiment Design
    ─────────────────────────────────
    🧪 Designing experiment to test hypothesis...
    📋 Success criteria: Sharpe > 1.0, IC > 0.05

    Phase 3: Code Generation (CoSTEER)
    ─────────────────────────────────
    💻 Generating factor implementation...
    ⚙️  Self-testing and evolving code...

    Phase 4: Execution
    ─────────────────────────────────
    ▶️  Running backtest in LocalEnv...
    📊 Results: Sharpe 1.23, IC 0.067

    Phase 5: Feedback
    ─────────────────────────────────
    ✅ Hypothesis validated!
    🔄 Suggestions for next iteration...
    """)

    print("💡 This is what runs with a real LLM API key!")

def main():
    print("\n" + "🌩️ " * 30)
    print("  RD-Agent Cloud Session Demo")
    print(f"  Session: {SESSION_ID}")
    print("  Running without Docker in Anthropic Cloud")
    print("🌩️ " * 30)

    has_real_llm = check_environment()

    # Always run LocalEnv demo
    run_simple_demo()

    if not has_real_llm:
        # Show mock demo
        run_factor_demo()

        print("\n" + "=" * 60)
        print("🔑 To Run Full RD-Agent:")
        print("=" * 60)
        print("""
1. Get an LLM API key:
   • OpenAI: https://platform.openai.com/api-keys
   • Azure: Use Azure OpenAI endpoint
   • DeepSeek: https://platform.deepseek.com (cheaper)

2. Set your API key:
   export OPENAI_API_KEY="sk-your-actual-key-here"

3. Run a real scenario:
   # Simple factor research
   python3 -c "from rdagent.app.qlib_rd_loop.factor import main; main(loop_n=3)"

   # Or use our 2025 demo with real execution
   python3 demo_quant_2025.py

4. Monitor progress:
   # In another terminal
   rdagent ui --port 19899 --log-dir log/
        """)
    else:
        print("\n✅ Ready to run full RD-Agent!")
        print("   Try: python3 demo_quant_2025.py")

if __name__ == "__main__":
    main()
PYEOF

chmod +x run_cloud_demo.py

echo "✓ Created run_cloud_demo.py"

# Create quick start guide
echo ""
echo "📚 Creating quick start guide..."

cat > CLOUD_QUICKSTART.md << 'MDEOF'
# RD-Agent Cloud Session Quick Start

## 🌩️ Running in Anthropic Cloud Session

This environment has been configured to run RD-Agent without Docker.

## Quick Demo (No API Key Needed)

```bash
# Run the cloud demo
python3 run_cloud_demo.py
```

This will:
- ✓ Check your environment
- ✓ Run a LocalEnv test
- ✓ Show what RD-Agent does with real LLM

## Full RD-Agent (Requires API Key)

### 1. Get an API Key

**OpenAI (Recommended)**
- Go to: https://platform.openai.com/api-keys
- Create new key
- Cost: ~$1-5 per experiment

**DeepSeek (Cheaper Alternative)**
- Go to: https://platform.deepseek.com
- Create API key
- Cost: ~10x cheaper than GPT-4

### 2. Configure

```bash
# Set your API key
export OPENAI_API_KEY="sk-your-actual-key-here"

# Verify configuration
python3 -c "import os; print('API Key:', os.getenv('OPENAI_API_KEY')[:20] + '...')"
```

### 3. Run Real Scenarios

**Option A: Our 2025 Quantitative Strategy Demo**
```bash
# Complete working backtest with ML
python3 demo_quant_2025.py
```

**Option B: Factor Research Loop**
```bash
# Run 3 iterations of factor discovery
python3 -m rdagent.app.qlib_rd_loop.factor --loop_n 3
```

**Option C: Feature Engineering Demo**
```bash
# Automated ML feature engineering
python3 demo_rd_loop.py
```

## What Works in Cloud Session

✅ **LocalEnv execution** - Runs code directly on host
✅ **All demos** - demo_quant_2025.py, demo_rd_loop.py, etc.
✅ **LLM-powered R&D** - Full hypothesis → code → eval loop
✅ **Synthetic data** - Generate and test on mock data

❌ **Docker scenarios** - Need native Docker daemon
❌ **Real market data** - Qlib data needs setup
❌ **Production deployment** - Use proper environment

## Example: 3-Iteration Factor Research

```bash
# Export your API key
export OPENAI_API_KEY="sk-..."
export CHAT_MODEL="gpt-4o"

# Run factor research (3 iterations)
python3 -m rdagent.app.qlib_rd_loop.factor --loop_n 3

# Results saved to:
# log/__session__/*/
```

## Monitor Progress

```bash
# In another terminal/tab
rdagent ui --port 19899 --log-dir log/
```

Then open: http://localhost:19899

## Cost Estimates

| Scenario | Iterations | Time | Cost (GPT-4) |
|----------|-----------|------|--------------|
| Simple demo | 1 | 2 min | $0 |
| Factor research | 3 | 10 min | $1-3 |
| Factor research | 10 | 30 min | $5-15 |
| Full pipeline | 20+ | 1+ hr | $20-50 |

💡 **Tip:** Use DeepSeek for ~10x cost savings

## Troubleshooting

**"No module named X"**
```bash
pip install pandas numpy scikit-learn dill
```

**"Docker not available"**
- Expected! Cloud session uses LocalEnv instead
- All demos work without Docker

**"API rate limit exceeded"**
- Wait a few minutes
- Or set: `RETRY_WAIT_SECONDS=60`

**"No valid API key"**
```bash
# Check if set
echo $OPENAI_API_KEY

# Should start with 'sk-'
export OPENAI_API_KEY="sk-your-key-here"
```

## Files Created

- `.env` - Environment configuration
- `run_cloud_demo.py` - Quick demo script
- `log/` - Execution logs
- `git_ignore_folder/` - Workspace

## Next Steps

1. **Run quick demo:** `python3 run_cloud_demo.py`
2. **Add API key:** `export OPENAI_API_KEY="sk-..."`
3. **Run real demo:** `python3 demo_quant_2025.py`
4. **Explore RD-Agent:** See README.md and documentation

---

Need help? Check:
- Main README: `/home/user/RD-Agent/README.md`
- Setup guide: `SETUP_GUIDE.md`
- Security audit: `SECURITY_AUDIT_REPORT.md`
MDEOF

echo "✓ Created CLOUD_QUICKSTART.md"

# Summary
echo ""
echo "=" "=========================================="
echo "✅ Cloud Session Setup Complete!"
echo "=========================================="
echo ""
echo "📁 Files created:"
echo "   • .env (cloud-optimized configuration)"
echo "   • run_cloud_demo.py (quick demo runner)"
echo "   • CLOUD_QUICKSTART.md (full guide)"
echo ""
echo "🚀 Quick Start:"
echo ""
echo "   1. Run demo with session ID:"
echo "      python3 run_cloud_demo.py $SESSION_ID"
echo ""
echo "   2. Or set as environment variable:"
echo "      export CLOUD_SESSION_ID=$SESSION_ID"
echo "      python3 run_cloud_demo.py"
echo ""
echo "   3. For full RD-Agent, add your API key:"
echo "      export OPENAI_API_KEY='sk-your-actual-key-here'"
echo ""
echo "   4. Then run a real scenario:"
echo "      python3 demo_quant_2025.py"
echo ""
echo "📚 Full guide: cat CLOUD_QUICKSTART.md"
echo ""
