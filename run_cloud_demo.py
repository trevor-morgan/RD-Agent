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
