#!/usr/bin/env python3
"""
Test RD-Agent LocalEnv - Run code without Docker
Demonstrates that RD-Agent can execute code locally
"""

import sys
from pathlib import Path

# Add rdagent to path
sys.path.insert(0, str(Path(__file__).parent))

from rdagent.utils.env import LocalConf, LocalEnv

def test_local_execution():
    """Test that LocalEnv can execute Python code directly"""

    print("=" * 60)
    print("🧪 Testing RD-Agent LocalEnv (No Docker Required)")
    print("=" * 60)

    # Create a simple Python script to execute
    test_workspace = Path("/tmp/rdagent_test_workspace")
    test_workspace.mkdir(exist_ok=True)

    test_script = test_workspace / "hello.py"
    test_script.write_text("""
print("Hello from RD-Agent LocalEnv!")
print("Running without Docker!")

# Test some computation
result = sum(range(1, 101))
print(f"Sum of 1-100: {result}")

# Test data structure
data = {"framework": "RD-Agent", "mode": "LocalEnv", "docker": False}
print(f"Config: {data}")
""")

    # Configure LocalEnv
    config = LocalConf(default_entry="python3")
    env = LocalEnv(conf=config)

    print("\n📝 Configuration:")
    print(f"   Workspace: {test_workspace}")
    print(f"   Script: {test_script.name}")
    print(f"   Mode: LocalEnv (no Docker)")

    # Execute the script
    print("\n▶️  Executing Python script locally...\n")

    try:
        result = env.run(
            entry=f"python3 {test_script}",
            local_path=str(test_workspace)
        )

        # Handle EnvResult object
        stdout = result.stdout if hasattr(result, 'stdout') else str(result)
        exit_code = result.exit_code if hasattr(result, 'exit_code') else 0

        print("📤 Output:")
        print("-" * 60)
        print(stdout)
        print("-" * 60)

        if exit_code == 0:
            print("\n✅ Success! LocalEnv executed code without Docker")
            print("\n🎯 Key Points:")
            print("   • No Docker daemon required")
            print("   • Runs directly on host system")
            print("   • Faster execution (no container overhead)")
            print("   • Works in this cloud session!")
        else:
            print(f"\n❌ Execution failed with exit code: {exit_code}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nThis might be due to missing dependencies or environment issues.")
        return False

    # Cleanup
    test_script.unlink()

    print("\n" + "=" * 60)
    print("✨ LocalEnv test completed!")
    print("=" * 60)

    return exit_code == 0

def test_environment_info():
    """Display environment information"""

    print("\n" + "=" * 60)
    print("🔍 Environment Information")
    print("=" * 60)

    import platform
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")

    # Check what's available
    checks = {
        "RD-Agent": False,
        "LocalEnv": False,
        "CondaEnv": False,
        "Docker": False
    }

    try:
        import rdagent
        checks["RD-Agent"] = True
        print(f"RD-Agent version: {rdagent.__version__ if hasattr(rdagent, '__version__') else 'dev'}")
    except ImportError:
        pass

    try:
        from rdagent.utils.env import LocalEnv
        checks["LocalEnv"] = True
    except ImportError:
        pass

    try:
        from rdagent.utils.env import CondaEnv
        checks["CondaEnv"] = True
    except ImportError:
        pass

    try:
        import docker
        client = docker.from_env()
        client.ping()
        checks["Docker"] = True
    except:
        checks["Docker"] = False

    print("\n📦 Available Components:")
    for component, available in checks.items():
        status = "✅" if available else "❌"
        print(f"   {status} {component}")

    print("\n💡 Recommendation:")
    if checks["LocalEnv"]:
        print("   Use LocalEnv for quick testing (no Docker needed)")
    if checks["CondaEnv"]:
        print("   Use CondaEnv for better isolation")
    if not checks["Docker"]:
        print("   ⚠️  Docker unavailable - limited to LocalEnv/CondaEnv")

if __name__ == "__main__":
    print("\n" + "🚀" * 30)
    print("  RD-Agent LocalEnv Test Suite")
    print("  (Demonstrating Docker-free execution)")
    print("🚀" * 30 + "\n")

    # Show environment info
    test_environment_info()

    # Run test
    success = test_local_execution()

    if success:
        print("\n✨ All tests passed! RD-Agent works without Docker in this environment.")
    else:
        print("\n⚠️  Tests encountered issues. See output above for details.")

    sys.exit(0 if success else 1)
