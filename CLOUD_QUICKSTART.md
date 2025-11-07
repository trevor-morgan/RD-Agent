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
