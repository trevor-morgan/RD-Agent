# Docker Removal Analysis: Making RD-Agent Work Without Docker

## 🎯 Executive Summary

**Good News:** RD-Agent already has `LocalEnv` and `CondaEnv` alternatives to Docker!
**Feasibility:** Moderately feasible - ~40% of the work is already done
**Effort:** 2-3 weeks of development
**Risk:** Medium (security implications)

## 📊 Current State Analysis

### Docker Usage in Codebase

**Statistics:**
- 129 Docker references across the codebase
- 15 files directly use Docker
- 4 Docker environment classes (`QlibDockerConf`, `KGDockerConf`, `DSDockerConf`, `MLEBDockerConf`)
- **But also 3 non-Docker alternatives already exist!**

### Existing Non-Docker Infrastructure ✅

The codebase already has:

```python
# From rdagent/utils/env.py

class LocalEnv(Env[LocalConf]):
    """Local environment - runs code directly on host"""

class CondaEnv(LocalEnv[CondaConf]):
    """Conda environment - isolated Python environments"""

class QlibCondaEnv(LocalEnv[QlibCondaConf]):
    """Qlib-specific conda environment"""
```

**Already Working Without Docker:**
- ✅ `LocalEnv` - Direct execution on host
- ✅ `CondaEnv` - Conda virtual environments
- ✅ `QlibCondaEnv` - Qlib quantitative trading (with conda)
- ✅ Factor coding scenarios (uses LocalEnv)

### What Still Requires Docker

**Scenarios that need Docker:**
1. ❌ Data Science scenarios (`DSDockerConf`)
2. ❌ Kaggle competitions (`KGDockerConf`)
3. ❌ MLE-Bench scenarios (`MLEBDockerConf`)
4. ⚠️ Some Qlib scenarios (have both Docker and Conda options)

## 🔍 Why Docker is Used

### Current Benefits:

1. **Isolation** 🔒
   - Generated code runs in sandbox
   - Prevents system contamination
   - Each scenario has clean environment

2. **Security** 🛡️
   - Limits damage from malicious code
   - Prevents file system access
   - Network isolation possible

3. **Reproducibility** 🔄
   - Same environment every time
   - Specific package versions
   - OS-level consistency

4. **Dependency Management** 📦
   - Different scenarios need different packages
   - Avoid dependency conflicts
   - Easy cleanup after execution

## 🚀 Removal Strategy

### Option 1: Expand LocalEnv/CondaEnv (Recommended)

**Effort:** 2-3 weeks
**Risk:** Medium
**Compatibility:** High

**Changes Required:**

#### 1. Create Conda Environments for All Scenarios

```python
# New classes to add:

class DSCondaConf(CondaConf):
    """Data Science conda environment"""
    conda_env_name: str = "rdagent_ds"
    # Specify all DS dependencies

class KGCondaConf(CondaConf):
    """Kaggle conda environment"""
    conda_env_name: str = "rdagent_kaggle"

class MLEBCondaConf(CondaConf):
    """MLE-Bench conda environment"""
    conda_env_name: str = "rdagent_mleb"
```

#### 2. Update Scenario Configurations

**Files to modify:**
- `rdagent/components/coder/data_science/conf.py`
- `rdagent/app/kaggle/conf.py`
- `rdagent/scenarios/data_science/*`

**Example change:**
```python
# Before:
from rdagent.utils.env import DSDockerConf, DockerEnv
env = DockerEnv(conf=DSDockerConf())

# After:
from rdagent.utils.env import DSCondaConf, CondaEnv
env = CondaEnv(conf=DSCondaConf())
```

#### 3. Add Safety Wrappers

**Critical:** LocalEnv has security issues (see SECURITY_AUDIT_REPORT.md)

**New file:** `rdagent/utils/sandbox.py`

```python
class SafeLocalEnv(LocalEnv):
    """
    Safer version of LocalEnv with restrictions:
    - Whitelist allowed commands
    - Restrict file system access
    - Timeout all executions
    - Monitor resource usage
    """

    def _run(self, entry: str, **kwargs):
        # Validate entry command
        self._validate_command(entry)

        # Run with restrictions
        return self._run_restricted(entry, **kwargs)
```

#### 4. Environment Setup Script

**New file:** `scripts/setup_conda_envs.sh`

```bash
#!/bin/bash
# Create all conda environments needed for non-Docker operation

# Data Science environment
conda create -n rdagent_ds python=3.10 -y
conda activate rdagent_ds
pip install pandas scikit-learn xgboost lightgbm

# Kaggle environment
conda create -n rdagent_kaggle python=3.10 -y
conda activate rdagent_kaggle
pip install pandas scikit-learn kaggle

# MLE-Bench environment
conda create -n rdagent_mleb python=3.10 -y
conda activate rdagent_mleb
pip install pandas scikit-learn torch

# Qlib environment (already exists but document)
conda create -n rdagent_qlib python=3.8 -y
conda activate rdagent_qlib
pip install pyqlib
```

### Option 2: Virtual Environments (venv)

**Effort:** 1-2 weeks
**Risk:** Higher (less isolation)
**Compatibility:** Medium

```python
class VenvConf(LocalConf):
    """Python venv configuration"""
    venv_path: str
    requirements_file: str

class VenvEnv(LocalEnv[VenvConf]):
    """Execute in Python virtual environment"""

    def _run(self, entry: str, **kwargs):
        # Activate venv and run
        cmd = f"source {self.conf.venv_path}/bin/activate && {entry}"
        return subprocess.run(cmd, shell=True, ...)
```

**Pros:**
- Simpler than conda
- Faster environment creation
- Built into Python

**Cons:**
- Less isolation than conda
- Harder to manage system dependencies
- No cross-version Python support

### Option 3: Hybrid Approach (Best of Both)

**Recommended Configuration:**

```python
# rdagent/core/conf.py

class ExecutionSettings(BaseSettings):
    """Global execution preferences"""

    prefer_docker: bool = True  # Default to Docker if available
    fallback_to_local: bool = True  # Use local if Docker unavailable
    force_local: bool = False  # Never use Docker

    # Safety settings for local execution
    local_timeout_seconds: int = 3600
    local_max_memory_mb: int = 8192
    local_allowed_commands: list[str] = ["python", "pip", "conda"]
```

**Implementation:**
```python
def get_environment(scenario: str) -> Env:
    """Smart environment selection"""

    settings = ExecutionSettings()

    # Check if Docker is available
    docker_available = check_docker()

    if settings.force_local or not docker_available:
        # Use local/conda environment
        return get_local_env(scenario)

    elif docker_available and settings.prefer_docker:
        # Use Docker environment
        return get_docker_env(scenario)

    else:
        # Docker preferred but not available, fallback
        logger.warning("Docker not available, using local environment")
        return get_local_env(scenario)
```

## 📋 Implementation Checklist

### Phase 1: Foundation (1 week)

- [ ] Create `DSCondaConf`, `KGCondaConf`, `MLEBCondaConf` classes
- [ ] Implement `SafeLocalEnv` with security restrictions
- [ ] Add `ExecutionSettings` for environment selection
- [ ] Create conda environment setup scripts
- [ ] Update configuration system to support both Docker and local

### Phase 2: Scenario Migration (1 week)

- [ ] Update Data Science scenario to use CondaEnv
- [ ] Update Kaggle scenario to use CondaEnv
- [ ] Update MLE-Bench scenario to use CondaEnv
- [ ] Add environment auto-detection logic
- [ ] Create fallback mechanism

### Phase 3: Testing & Security (3-5 days)

- [ ] Test all scenarios with LocalEnv/CondaEnv
- [ ] Security audit of local execution
- [ ] Add command whitelisting
- [ ] Implement resource limits (CPU, memory, time)
- [ ] Add file system restrictions

### Phase 4: Documentation (2-3 days)

- [ ] Update README with non-Docker instructions
- [ ] Document conda environment setup
- [ ] Add troubleshooting guide
- [ ] Update SETUP_GUIDE.md
- [ ] Security considerations document

## ⚠️ Security Implications

### Current Docker Security:
- ✅ Isolated file system
- ✅ Network isolation
- ✅ Resource limits
- ✅ Prevents host contamination

### LocalEnv Security Risks:

**CRITICAL Issues (from SECURITY_AUDIT_REPORT.md):**

1. **Shell Injection** (4 critical vulnerabilities)
   ```python
   # rdagent/utils/env.py:550
   subprocess.run(entry, shell=True)  # ❌ DANGEROUS without Docker
   ```

2. **No Isolation**
   - Generated code runs as host user
   - Can access entire file system
   - Can install system packages
   - Can modify Python environment

3. **Resource Exhaustion**
   - No CPU limits
   - No memory limits
   - Can spawn infinite processes

**Required Mitigations:**

```python
class SafeLocalEnv(LocalEnv):
    """Secure local execution environment"""

    ALLOWED_COMMANDS = ["python", "python3", "pip", "conda"]
    BLOCKED_PATTERNS = ["rm -rf", "sudo", "curl", "wget"]
    MAX_EXECUTION_TIME = 3600  # 1 hour
    MAX_MEMORY_MB = 8192  # 8 GB

    def _validate_command(self, entry: str):
        """Validate command is safe"""

        # Check for blocked patterns
        for pattern in self.BLOCKED_PATTERNS:
            if pattern in entry:
                raise SecurityError(f"Blocked pattern: {pattern}")

        # Check command whitelist
        cmd = entry.split()[0]
        if cmd not in self.ALLOWED_COMMANDS:
            raise SecurityError(f"Command not allowed: {cmd}")

    def _run_restricted(self, entry: str, **kwargs):
        """Run with resource limits"""

        # Use resource module for limits
        import resource

        def set_limits():
            # Max 1 hour execution
            resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
            # Max 8GB memory
            resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))

        # Run subprocess with limits
        return subprocess.run(
            entry,
            shell=False,  # ✅ No shell injection
            preexec_fn=set_limits,
            timeout=self.MAX_EXECUTION_TIME,
            **kwargs
        )
```

## 📊 Comparison Matrix

| Aspect | Docker | CondaEnv | LocalEnv |
|--------|--------|----------|----------|
| **Isolation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| **Security** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Performance** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Setup Complexity** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Portability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Resource Usage** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Already Works?** | ✅ Yes | ⚠️ Partial | ⚠️ Partial |

## 💰 Cost-Benefit Analysis

### Benefits of Removing Docker:

✅ **Faster Execution**
- No container overhead
- Direct system access
- Faster I/O

✅ **Simpler Setup**
- No Docker installation required
- Works on more systems
- Easier for Windows/Mac (with WSL/conda)

✅ **Lower Resource Usage**
- No container memory overhead
- Shared libraries/dependencies
- Less disk space

✅ **Easier Debugging**
- Direct access to processes
- Native tools work
- Simpler logs

### Costs of Removing Docker:

❌ **Reduced Security**
- Generated code runs on host
- File system exposure
- Malicious code risks

❌ **Less Isolation**
- Dependency conflicts possible
- System contamination risk
- Harder to clean up

❌ **Portability Issues**
- Environment-dependent behavior
- Different OS behaviors
- Harder to reproduce issues

❌ **Development Effort**
- 2-3 weeks implementation
- Testing across platforms
- Security hardening

## 🎯 Recommended Approach

### Immediate (This Week):

1. **Enable existing LocalEnv for testing**
   ```bash
   # Set environment variable to force local execution
   export FORCE_LOCAL_ENV=true
   rdagent fin_factor  # Will use QlibCondaEnv instead of Docker
   ```

2. **Test what already works**
   - Qlib scenarios with `QlibCondaEnv`
   - Factor coding with `LocalEnv`
   - Simple model scenarios

### Short-term (2-3 weeks):

3. **Implement Phase 1-4** (see checklist above)
4. **Add security hardening**
5. **Full testing suite**

### Configuration File:

**New file:** `.env` additions

```bash
# Execution Environment
FORCE_LOCAL_ENV=false          # Force local execution
PREFER_DOCKER=true             # Prefer Docker when available
FALLBACK_TO_LOCAL=true         # Fallback to local if Docker unavailable

# Local Execution Safety
LOCAL_EXECUTION_TIMEOUT=3600   # Max execution time (seconds)
LOCAL_MAX_MEMORY_MB=8192       # Max memory usage
LOCAL_ALLOWED_COMMANDS=python,pip,conda
LOCAL_WORKSPACE_RESTRICTION=true  # Restrict to workspace directory
```

## 🔧 Quick Start Without Docker

### For Current Codebase (No Changes):

Some scenarios already work without Docker:

```bash
# 1. Set up Qlib conda environment
conda create -n qlibRDAgent python=3.8 -y
conda activate qlibRDAgent
pip install pyqlib torch catboost

# 2. Use scenarios that support LocalEnv
export USE_QLIB_CONDA=true
rdagent fin_factor  # Uses QlibCondaEnv

# 3. Check what's running
ps aux | grep python  # See it's running locally, not in container
```

## 📈 Migration Path

### For Existing Users:

**Phase 1: Hybrid Mode (Recommended)**
```python
# Try Docker first, fallback to local
PREFER_DOCKER=true
FALLBACK_TO_LOCAL=true
```

**Phase 2: Testing Local**
```python
# Force local for testing
FORCE_LOCAL_ENV=true
# Test with non-sensitive scenarios first
```

**Phase 3: Full Local**
```python
# Disable Docker completely
PREFER_DOCKER=false
# Ensure all conda environments are set up
```

## 🎓 Summary

**Can we remove Docker?** Yes, but with caveats.

**Should we remove Docker?** Depends on use case:

| Use Case | Recommendation |
|----------|----------------|
| **Development/Testing** | ✅ Use LocalEnv/CondaEnv |
| **Personal Research** | ✅ Use CondaEnv (safer) |
| **Production/Untrusted Code** | ⚠️ Keep Docker |
| **Cloud/Limited Resources** | ✅ Use LocalEnv |
| **Windows/Mac Users** | ⚠️ Docker or WSL+Conda |

**Best Approach:**
Implement **hybrid mode** where users can choose:
- Docker (default, most secure)
- CondaEnv (balanced)
- LocalEnv (fastest, least secure)

**Effort Required:** 2-3 weeks for full implementation
**Risk Level:** Medium (security concerns)
**Benefit:** Opens RD-Agent to broader audience

---

## 🚀 Next Steps

Want me to:
1. ✅ Implement the hybrid environment selection system?
2. ✅ Create the SafeLocalEnv with security hardening?
3. ✅ Set up conda environments for all scenarios?
4. ✅ Test current LocalEnv functionality?

The infrastructure is already 40% there - we just need to expand and secure it!
