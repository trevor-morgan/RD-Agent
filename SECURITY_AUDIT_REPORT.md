# RD-Agent Codebase Security Vulnerability Audit Report

## Executive Summary
This comprehensive security audit of the RD-Agent codebase identified **18 security vulnerabilities** across multiple categories:
- 6 Command Injection issues (Critical/High severity)
- 3 Unsafe Deserialization issues (High/Medium severity)  
- 3 Environment Variable Exposure issues (Medium severity)
- 2 Docker Security issues (Medium severity)
- 2 Input Validation issues (Medium/Low severity)
- 2 Path Traversal risks (Low severity)

**Overall Risk Assessment: HIGH** - Multiple critical command injection vulnerabilities found

---

## Detailed Findings

### Category 1: COMMAND INJECTION (Critical/High Severity)

#### Issue 1.1: Shell Injection in LocalEnv._run()
**File**: `/home/user/RD-Agent/rdagent/utils/env.py`
**Lines**: 550, 620
**Severity**: CRITICAL

**Vulnerability Description**:
The `LocalEnv._run()` method uses `shell=True` with user-controlled `entry` parameter. This allows arbitrary shell command execution.

```python
# Line 550
process = subprocess.Popen(
    entry,  # User-controlled input from self.conf.default_entry
    cwd=cwd,
    env={**os.environ, **env},
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    shell=True,  # DANGEROUS: allows command injection
    bufsize=1,
    universal_newlines=True,
)
```

**Attack Example**:
```python
env = LocalEnv(LocalConf(default_entry="echo test; rm -rf /"))
env.run()  # Would execute arbitrary commands
```

**Fix**:
```python
# Use list-based command parsing instead of shell=True
import shlex
command_parts = shlex.split(entry)
process = subprocess.Popen(
    command_parts,
    cwd=cwd,
    env={**os.environ, **env},
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    shell=False,  # Disable shell interpretation
    bufsize=1,
    universal_newlines=True,
)
```

---

#### Issue 1.2: Shell Injection in CondaConf.change_bin_path()
**File**: `/home/user/RD-Agent/rdagent/utils/env.py`
**Lines**: 616-621
**Severity**: HIGH

**Vulnerability Description**:
Uses `shell=True` with f-string interpolation including user-controlled `conda_env_name`.

```python
# Line 616-621
conda_path_result = subprocess.run(
    f"conda run -n {self.conda_env_name} --no-capture-output env | grep '^PATH='",
    capture_output=True,
    text=True,
    shell=True,  # DANGEROUS with f-string
)
```

**Attack Example**:
```python
conf = CondaConf(conda_env_name="test; rm -rf /tmp/*; echo")
# Would execute: "conda run -n test; rm -rf /tmp/*; echo --no-capture-output env | grep '^PATH='"
```

**Fix**:
```python
command = ["conda", "run", "-n", self.conda_env_name, "--no-capture-output", "env"]
grep_command = ["grep", "^PATH="]
conda_result = subprocess.run(command, capture_output=True, text=True)
conda_path_result = subprocess.run(
    grep_command,
    input=conda_result.stdout,
    capture_output=True,
    text=True,
)
```

---

#### Issue 1.3: Shell Injection in FactorFBWorkspace.execute()
**File**: `/home/user/RD-Agent/rdagent/components/coder/factor_coder/factor.py`
**Lines**: 163-169
**Severity**: CRITICAL

**Vulnerability Description**:
F-string command construction with user-controlled path variables passed to shell=True.

```python
# Lines 163-169
subprocess.check_output(
    f"{FACTOR_COSTEER_SETTINGS.python_bin} {execution_code_path}",
    shell=True,  # DANGEROUS: f-string interpolation
    cwd=self.workspace_path,
    stderr=subprocess.STDOUT,
    timeout=FACTOR_COSTEER_SETTINGS.file_based_execution_timeout,
)
```

**Risk**: If `execution_code_path` contains spaces or special characters, command injection is possible.

**Fix**:
```python
subprocess.check_output(
    [FACTOR_COSTEER_SETTINGS.python_bin, str(execution_code_path)],
    cwd=self.workspace_path,
    stderr=subprocess.STDOUT,
    timeout=FACTOR_COSTEER_SETTINGS.file_based_execution_timeout,
    shell=False,  # Use list format, no shell
)
```

---

#### Issue 1.4: Shell Injection in QlibCondaEnv.prepare()
**File**: `/home/user/RD-Agent/rdagent/utils/env.py`
**Lines**: 674-692
**Severity**: CRITICAL

**Vulnerability Description**:
Multiple `subprocess` calls with `shell=True` and f-string interpolation, including user-controlled `conda_env_name`.

```python
# Lines 677-679
subprocess.check_call(
    f"conda create -y -n {self.conf.conda_env_name} python=3.10",
    shell=True,  # DANGEROUS
)

# Lines 681-683
subprocess.check_call(
    f"conda run -n {self.conf.conda_env_name} pip install --upgrade pip cython",
    shell=True,  # DANGEROUS
)
```

**Attack**: Malicious conda environment names could execute arbitrary commands.

**Fix**:
```python
subprocess.check_call(
    ["conda", "create", "-y", "-n", self.conf.conda_env_name, "python=3.10"],
    shell=False,
)

subprocess.check_call(
    ["conda", "run", "-n", self.conf.conda_env_name, "pip", "install", "--upgrade", "pip", "cython"],
    shell=False,
)
```

---

#### Issue 1.5: Command Injection in DS Loop Shell Entry Point
**File**: `/home/user/RD-Agent/rdagent/utils/env.py`
**Lines**: 305-331
**Severity**: HIGH

**Vulnerability Description**:
Shell command construction with untrusted input (`find_cmd` and `chmod_cmd` are built with string concatenation).

```python
# Lines 305-311
find_cmd = f"find {workspace_path} -mindepth 1 -maxdepth 1"
for name in [...]:
    find_cmd += f" ! -name {name}"  # Unsanitized name concatenation
chmod_cmd = f"{find_cmd} -exec chmod -R 777 {{}} +"

# Lines 319-330
entry_add_timeout = (
    f"/bin/sh -c '"
    + f"{timeout_cmd}; entry_exit_code=$?; "
    + f"{_get_chmod_cmd(self.conf.mount_path)}; "
    + "exit $entry_exit_code'"
)
```

**Risk**: If `name` variables contain shell metacharacters, command injection occurs.

**Fix**:
```python
import subprocess
import shlex

# Build a proper find command without shell
find_args = ["find", workspace_path, "-mindepth", "1", "-maxdepth", "1"]
for name in [...]:
    find_args.extend(["!", "-name", name])  # Arguments properly separated

# Use subprocess directly instead of building shell strings
subprocess.run(find_args + ["-exec", "chmod", "-R", "777", "{}", "+"])
```

---

#### Issue 1.6: Unquoted Path in CLI Streamlit Subprocess
**File**: `/home/user/RD-Agent/rdagent/app/cli.py`
**Lines**: 43, 47, 61, 69
**Severity**: MEDIUM

**Vulnerability Description**:
While using list-based `subprocess.run()` (which is safer), paths from `rpath()` could contain spaces if not properly quoted.

```python
# Lines 42-44
with rpath("rdagent.log.ui", "app.py") as app_path:
    cmds = ["streamlit", "run", app_path, f"--server.port={port}"]
    subprocess.run(cmds)
```

**Issue**: If `app_path` contains spaces, Streamlit might not receive it correctly. However, this is less critical because list-based subprocess is used.

**Fix**:
```python
# Ensure app_path is properly converted
with rpath("rdagent.log.ui", "app.py") as app_path:
    cmds = ["streamlit", "run", str(app_path), f"--server.port={port}"]
    subprocess.run(cmds)
```

---

### Category 2: UNSAFE DESERIALIZATION (High/Medium Severity)

#### Issue 2.1: Unsafe Pickle Loading from User-Controlled Cache
**File**: `/home/user/RD-Agent/rdagent/utils/env.py`
**Lines**: 383-384
**Severity**: HIGH

**Vulnerability Description**:
Pickle objects loaded from cache without validation. If cache is compromised, arbitrary code execution occurs.

```python
# Lines 382-384
if Path(target_folder / f"{key}.pkl").exists() and Path(target_folder / f"{key}.zip").exists():
    with open(target_folder / f"{key}.pkl", "rb") as f:
        ret = pickle.load(f)  # UNSAFE: No validation
```

**Risk**: Pickle can deserialize arbitrary Python objects and execute code.

**Fix**:
```python
# Use safer serialization methods
import json

# Instead of pickle:
with open(target_folder / f"{key}.json", "r") as f:
    ret = json.load(f)  # JSON is safer - only deserializes data, not code

# If pickle is necessary, use restricted unpickler:
import pickle
import io

class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Whitelist allowed classes
        ALLOWED_MODULES = {"pathlib", "collections"}
        if module.split(".")[0] in ALLOWED_MODULES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Class {module}.{name} not allowed")

with open(target_folder / f"{key}.pkl", "rb") as f:
    ret = RestrictedUnpickler(f).load()
```

---

#### Issue 2.2: Unsafe Pickle Loading in UI Components
**File**: `/home/user/RD-Agent/rdagent/log/ui/ds_user_interact.py`
**Lines**: 119-120, 138-139
**Severity**: HIGH

**Vulnerability Description**:
Multiple pickle.load() calls from user interaction folder without validation.

```python
# Lines 119-120
session_data = pickle.load(
    open(DS_RD_SETTING.user_interaction_mid_folder / f"{state.selected_session_name}.pkl", "rb")
)

# Lines 138-139
session_data = pickle.load(open(session_file, "rb"))
```

**Risk**: Attacker could craft malicious pickle files in the interaction folder.

**Fix**:
```python
# Use JSON instead of pickle for user interaction data
import json

try:
    with open(session_file, "r") as f:
        session_data = json.load(f)
except (json.JSONDecodeError, FileNotFoundError):
    continue
```

---

#### Issue 2.3: Unsafe Pickle Loading from Experiment Cache
**File**: `/home/user/RD-Agent/rdagent/core/knowledge_base.py`
**Lines**: 16
**Severity**: MEDIUM

**Vulnerability Description**:
Pickle loading from persistent cache without version/signature validation.

```python
# Line 16
loaded = pickle.load(f)  # No validation
```

**Fix**:
```python
import hmac
import hashlib

# Add integrity check
cache_file = Path(...)
signature_file = cache_file.with_suffix(".sig")

def verify_pickle_integrity(pickle_file, signature_file, key):
    if not signature_file.exists():
        raise ValueError("Signature file missing")
    
    with open(pickle_file, "rb") as f:
        data = f.read()
    
    with open(signature_file, "r") as f:
        stored_sig = f.read()
    
    computed_sig = hmac.new(key.encode(), data, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(computed_sig, stored_sig):
        raise ValueError("Pickle integrity check failed")
    
    return pickle.loads(data)

loaded = verify_pickle_integrity(cache_file, signature_file, "secret_key")
```

---

### Category 3: ENVIRONMENT VARIABLE EXPOSURE (Medium Severity)

#### Issue 3.1: Secrets Passed Through Environment Variables to Subprocesses
**File**: `/home/user/RD-Agent/rdagent/utils/env.py`
**Line**: 546
**Severity**: MEDIUM

**Vulnerability Description**:
All environment variables including sensitive credentials are passed to subprocesses.

```python
# Line 546
env={**os.environ, **env},  # Exposes all parent environment variables including secrets
```

**Risk**: Subprocess can access API keys, passwords, and other sensitive environment variables.

**Fix**:
```python
# Only pass required environment variables
safe_env = {
    k: v for k, v in os.environ.items() 
    if k in ["PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL"]
}
safe_env.update(env or {})

process = subprocess.Popen(
    entry,
    env=safe_env,  # Only necessary vars
    ...
)
```

---

#### Issue 3.2: API Keys in Environment Variable Example File
**File**: `/home/user/RD-Agent/.env.example`
**Lines**: 31-43
**Severity**: MEDIUM

**Vulnerability Description**:
Template file shows actual API key format (though with dummy values). Users might expose real keys.

```
OPENAI_API_KEY="sk-chat-key"  # Real format exposed
LITELLM_PROXY_API_KEY="sk-embedding-service-key"
```

**Risk**: Documentation pattern shows users how to expose credentials.

**Fix**:
```
# Better format:
# OPENAI_API_KEY="sk-..."  # Keep secret, never commit
# LITELLM_PROXY_API_KEY="sk-..."  # Keep secret, never commit
```

---

#### Issue 3.3: OS.environ Access for Sensitive Data
**File**: `/home/user/RD-Agent/rdagent/app/utils/health_check.py`
**Lines**: Multiple (see grep results)
**Severity**: MEDIUM

**Vulnerability Description**:
Direct access to environment variables for API keys without validation.

```python
# Health check accesses OPENAI_API_KEY, DEEPSEEK_API_KEY directly
if "DEEPSEEK_API_KEY" in os.environ:
    chat_api_key = os.getenv("DEEPSEEK_API_KEY")
```

**Risk**: Keys are held in memory and logged in error messages.

**Fix**:
```python
import os
from typing import Optional

def get_api_key(env_var: str) -> Optional[str]:
    """Safely retrieve and validate API keys."""
    key = os.getenv(env_var)
    if key:
        # Validate format
        if not key.startswith("sk-"):
            logger.warning(f"Suspicious {env_var} format")
        # Don't log the actual key
        logger.info(f"Using {env_var[0]}***{env_var[-4:]}")
    return key
```

---

### Category 4: DOCKER SECURITY (Medium Severity)

#### Issue 4.1: Using "latest" Tag for Base Images
**File**: `/home/user/RD-Agent/rdagent/scenarios/data_science/sing_docker/Dockerfile`
**Line**: 2
**Severity**: MEDIUM

**Vulnerability Description**:
Uses `pytorch/pytorch:latest` which can change unpredictably and introduce vulnerabilities.

```dockerfile
FROM pytorch/pytorch:latest  # DANGEROUS: Unpinned version
```

**Risk**: Automatic updates could introduce vulnerabilities or break reproducibility.

**Fix**:
```dockerfile
# Use specific version tags
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Or at least use specific year/month tags:
# FROM pytorch/pytorch:2024.01-cuda12.1-cudnn8-runtime
```

---

#### Issue 4.2: Git Clone from Branches Without Version Pinning
**File**: `/home/user/RD-Agent/rdagent/scenarios/kaggle/docker/mle_bench_docker/Dockerfile`
**Lines**: 22-25
**Severity**: MEDIUM

**Vulnerability Description**:
Clones Git repositories without specifying commits, allowing injection of malicious code.

```dockerfile
RUN git clone https://github.com/openai/mle-bench.git
RUN cd mle-bench && git lfs fetch --all
RUN cd mle-bench && git lfs pull
RUN cd mle-bench && python -m pip install -e .
```

**Risk**: Repository could be compromised or changed.

**Fix**:
```dockerfile
RUN git clone --depth 1 --branch v1.0.0 https://github.com/openai/mle-bench.git
RUN cd mle-bench && git verify-commit HEAD || exit 1  # Verify GPG signature
```

---

### Category 5: INPUT VALIDATION ISSUES (Medium/Low Severity)

#### Issue 5.1: Unvalidated User Input in Streamlit UI
**File**: `/home/user/RD-Agent/rdagent/log/ui/ds_user_interact.py`
**Lines**: 58-78
**Severity**: MEDIUM

**Vulnerability Description**:
User input from text_area() is directly stored in JSON without validation.

```python
# Lines 58-78
target_hypothesis = st.text_area(
    "Target hypothesis: (you can copy from candidates)",
    value=(original_hypothesis := selected_session_data["target_hypothesis"].hypothesis),
    height="content",
)
# No validation before storing
json.dump(
    submit_dict,  # Contains unvalidated user input
    open(DS_RD_SETTING.user_interaction_mid_folder / f"{state.selected_session_name}_RET.json", "w"),
)
```

**Risk**: XSS, injection attacks in JSON, code injection when hypothesis is parsed.

**Fix**:
```python
import re
from typing import Optional

def validate_hypothesis(hypothesis: str) -> Optional[str]:
    """Validate and sanitize user hypothesis input."""
    if not hypothesis or not isinstance(hypothesis, str):
        return None
    
    # Limit length
    if len(hypothesis) > 5000:
        st.error("Hypothesis too long (max 5000 chars)")
        return None
    
    # Check for suspicious patterns
    if re.search(r'[<>{}]|__', hypothesis):
        st.warning("Hypothesis contains suspicious characters")
        return None
    
    return hypothesis.strip()

target_hypothesis = validate_hypothesis(target_hypothesis)
if target_hypothesis is None:
    st.stop()
```

---

#### Issue 5.2: Session ID Validation in User Interaction
**File**: `/home/user/RD-Agent/rdagent/scenarios/data_science/interactor/__init__.py`
**Lines**: 98, 108
**Severity**: LOW

**Vulnerability Description**:
Session ID from user selection not validated before file operations.

```python
# Lines 98-108
session_id = uuid.uuid4().hex  # Generated safely
# But later:
Path(DS_RD_SETTING.user_interaction_mid_folder / f"{session_id}.pkl").unlink()
```

**Risk**: While uuid4 is safe, the pattern could be exploited if session_id comes from user input.

**Fix**:
```python
import uuid
import os

def validate_session_id(session_id: str) -> bool:
    """Validate session ID format."""
    try:
        uuid.UUID(session_id)
        return True
    except ValueError:
        return False

def safe_session_path(session_id: str, folder: Path) -> Path:
    """Safely construct session path with validation."""
    if not validate_session_id(session_id):
        raise ValueError("Invalid session ID")
    
    path = folder / f"{session_id}.pkl"
    
    # Verify path is within folder
    if not str(path.resolve()).startswith(str(folder.resolve())):
        raise ValueError("Path traversal detected")
    
    return path
```

---

### Category 6: PATH TRAVERSAL RISKS (Low Severity)

#### Issue 6.1: Dynamic Path Construction in File Operations
**File**: `/home/user/RD-Agent/rdagent/scenarios/data_science/interactor/__init__.py`
**Lines**: 98-112
**Severity**: LOW

**Vulnerability Description**:
While using UUID4 mitigates this, the pattern is vulnerable to path traversal if input validation is removed.

```python
# Lines 98-112 - Currently safe due to UUID4
session_id = uuid.uuid4().hex
Path(DS_RD_SETTING.user_interaction_mid_folder / f"{session_id}.pkl")

# But if session_id comes from untrusted source:
# Could be exploited with: "../../../etc/passwd"
```

**Fix**:
```python
from pathlib import Path
import os

def safe_path_join(base: Path, relative: str) -> Path:
    """Safely join paths preventing traversal."""
    base = base.resolve()
    full = (base / relative).resolve()
    
    # Check that result is still under base
    if not str(full).startswith(str(base)):
        raise ValueError(f"Path traversal detected: {relative}")
    
    return full

# Usage:
safe_path = safe_path_join(
    DS_RD_SETTING.user_interaction_mid_folder,
    f"{session_id}.pkl"
)
```

---

#### Issue 6.2: Volume Mount Path Construction in Docker
**File**: `/home/user/RD-Agent/rdagent/utils/env.py`
**Lines**: 74-93
**Severity**: LOW

**Vulnerability Description**:
Path normalization using os.path.join could allow mounting unintended directories.

```python
# Lines 77-79
def to_abs(path: str) -> str:
    return os.path.abspath(os.path.join(working_dir, path)) if not os.path.isabs(path) else path
```

**Risk**: If path contains `..`, could escape workspace.

**Fix**:
```python
from pathlib import Path
import os

def normalize_volumes(vols: dict, working_dir: str) -> dict:
    """Safely normalize volume paths."""
    abs_vols = {}
    working_path = Path(working_dir).resolve()
    
    for lp, vinfo in vols.items():
        if isinstance(vinfo, dict):
            vinfo = vinfo.copy()
            target = Path(vinfo["bind"])
            if not target.is_absolute():
                target = (working_path / target).resolve()
            
            # Verify no escape
            try:
                target.relative_to(working_path)
            except ValueError:
                logger.warning(f"Volume path escapes working directory: {target}")
                continue
            
            vinfo["bind"] = str(target)
            abs_vols[lp] = vinfo
        else:
            target = Path(vinfo)
            if not target.is_absolute():
                target = (working_path / target).resolve()
            abs_vols[lp] = str(target)
    
    return abs_vols
```

---

### Category 7: DEPENDENCY VULNERABILITIES (Informational)

#### Issue 7.1: Outdated/Vulnerable Dependencies
**File**: `/home/user/RD-Agent/requirements.txt`
**Severity**: MEDIUM

**Vulnerable Packages Identified**:
- `selenium` - Known vulnerabilities in older versions
- `docker` - Requires regular updates
- `litellm>=1.73` - Check for security updates
- `PyYAML` - No version pinning (could get vulnerable versions)
- `pandas`, `numpy` - Recommend explicit version pinning

**Risk**: Known vulnerabilities in dependencies could be exploited.

**Fix**:
```
# Pinned versions with known security status
selenium>=4.15.0  # Latest with security patches
docker>=7.0.0     # Requires security updates
litellm>=1.73,<1.80  # Pin to tested range
PyYAML>=6.0.1     # Requires version pinning
pandas>=2.0.0,<2.2
numpy>=1.24.0,<1.26
```

---

## Summary Table

| ID | Category | File | Severity | Line(s) | Quick Fix |
|----|----------|------|----------|---------|-----------|
| 1.1 | Command Injection | rdagent/utils/env.py | CRITICAL | 550 | Remove shell=True, use list-based args |
| 1.2 | Command Injection | rdagent/utils/env.py | HIGH | 616-621 | Use list format, no f-strings |
| 1.3 | Command Injection | rdagent/components/coder/factor_coder/factor.py | CRITICAL | 163-169 | Use list format, no shell |
| 1.4 | Command Injection | rdagent/utils/env.py | CRITICAL | 674-692 | Use list format for all subprocess calls |
| 1.5 | Command Injection | rdagent/utils/env.py | HIGH | 305-331 | Avoid shell command construction |
| 1.6 | Command Injection | rdagent/app/cli.py | MEDIUM | 43,47,61,69 | Ensure list format (already done) |
| 2.1 | Unsafe Pickle | rdagent/utils/env.py | HIGH | 383-384 | Use JSON or RestrictedUnpickler |
| 2.2 | Unsafe Pickle | rdagent/log/ui/ds_user_interact.py | HIGH | 119-120 | Switch to JSON |
| 2.3 | Unsafe Pickle | rdagent/core/knowledge_base.py | MEDIUM | 16 | Add integrity verification |
| 3.1 | Env Exposure | rdagent/utils/env.py | MEDIUM | 546 | Filter environment variables |
| 3.2 | Env Exposure | .env.example | MEDIUM | 31-43 | Use template comments |
| 3.3 | Env Exposure | rdagent/app/utils/health_check.py | MEDIUM | Multiple | Add key validation |
| 4.1 | Docker Security | sing_docker/Dockerfile | MEDIUM | 2 | Pin specific version tag |
| 4.2 | Docker Security | mle_bench_docker/Dockerfile | MEDIUM | 22-25 | Pin commit hash, verify |
| 5.1 | Input Validation | rdagent/log/ui/ds_user_interact.py | MEDIUM | 58-78 | Validate user input |
| 5.2 | Input Validation | rdagent/scenarios/data_science/interactor/__init__.py | LOW | 98,108 | Add session ID validation |
| 6.1 | Path Traversal | rdagent/scenarios/data_science/interactor/__init__.py | LOW | 98-112 | Implement safe path join |
| 6.2 | Path Traversal | rdagent/utils/env.py | LOW | 74-93 | Verify no path escape |

---

## Recommendations

### Immediate Actions (Critical)
1. **Fix all command injection issues (1.1-1.5)**: Replace all `shell=True` with list-based subprocess calls
2. **Remove unsafe pickle usage (2.1-2.3)**: Switch to JSON serialization or use RestrictedUnpickler
3. **Add input validation (5.1)**: Validate all user inputs before processing

### Short-term Actions (High Priority)
4. **Implement environment variable filtering**: Only pass necessary vars to subprocesses
5. **Update Docker images**: Pin specific versions, avoid "latest" tags
6. **Add path traversal protections**: Implement safe path operations

### Long-term Actions (Medium Priority)  
7. **Dependency scanning**: Regular security audits of requirements.txt
8. **Code security standards**: Implement pre-commit hooks for security checks
9. **Security testing**: Add SAST (Static Application Security Testing) to CI/CD

### Tools to Implement
```bash
# Pre-commit security checks
pip install bandit safety

# In .pre-commit-config.yaml:
- repo: https://github.com/PyCQA/bandit
  rev: 1.7.5
  hooks:
  - id: bandit
  
- repo: https://github.com/Lucas-C/pre-commit-hooks
  rev: v1.5.4
  hooks:
  - id: python-safety-dependencies-check
```

---

## Severity Scale

- **CRITICAL**: Immediate code execution or data breach possible
- **HIGH**: Easy exploitation possible, significant impact
- **MEDIUM**: Exploitation requires specific conditions, moderate impact
- **LOW**: Difficult exploitation, minimal impact

