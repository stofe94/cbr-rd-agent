# Local Installation Guide

> **CBR-Enhanced RD-Agent — Windows (WSL2) & Linux**
>
> This guide covers running the tool **directly without Docker**.
> Docker is still required internally — RD-Agent spawns containers for each
> experiment — but the agent itself runs in your local Python environment.

---

## Windows via WSL2

### Step 1 — Enable WSL2

Open PowerShell **as Administrator**:

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

**→ Restart Windows.**

Then set WSL2 as default:

```powershell
wsl --set-default-version 2
```

### Step 2 — Install Ubuntu

```powershell
wsl --install -d Ubuntu-22.04
```

On first launch you will be prompted for a username and password. Then update packages:

```bash
sudo apt update && sudo apt upgrade -y
```

### Step 3 — Install Docker Engine

#### Option A — Docker Engine directly in WSL2 (recommended)

```bash
sudo apt install -y ca-certificates curl gnupg lsb-release

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Allow running Docker without `sudo`:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

Auto-start Docker on WSL2 launch (WSL2 has no systemd by default):

```bash
# Add to ~/.bashrc
if [ "$(service docker status 2>&1 | grep -c "not running")" -eq 1 ]; then
  sudo service docker start
fi

# Allow passwordless start
echo "$USER ALL=(ALL) NOPASSWD: /usr/sbin/service docker start" | \
  sudo tee /etc/sudoers.d/docker-service
```

#### Option B — Docker Desktop for Windows

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and
enable the WSL2 backend in Settings → Resources → WSL Integration.

> ⚠️ If you previously used Docker Desktop and switched to native Docker Engine,
> remove the leftover credential helper from `~/.docker/config.json` — see
> Troubleshooting below.

#### Verify

```bash
docker run hello-world
```

---

## Linux (native)

Docker Engine installation follows the same steps as Option A above.
Skip the WSL2 steps entirely.

---

## Step 4 — Python Environment

[Miniforge3](https://github.com/conda-forge/miniforge) is recommended:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
source ~/.bashrc

conda create -n rdagent python=3.11 -y
conda activate rdagent
```

Or with `venv`:

```bash
python3 -m venv ~/envs/rdagent
source ~/envs/rdagent/bin/activate
```

---

## Step 5 — Clone & Install

```bash
git clone https://github.com/stofe94/cbr-rd-agent.git
cd cbr-rd-agent
pip install -e .
```

This installs the modified RD-Agent, reduced to the data science scenario,
including the CBR module and Google AI Studio backend. The `rdagent` command
will be available immediately after.

---

## Step 6 — Configuration

Copy the example files and fill in your values:

```bash
cp config.env.example config.env
cp secrets.env.example secrets.env
```

`secrets.env`:
```bash
GOOGLE_AI_STUDIO_API_KEY=...
KAGGLE_USERNAME=...
KAGGLE_KEY=...
```

---

## Step 7 — Run

```bash
bash run_linux.sh
```

`run_linux.sh` loads `config.env` automatically via `set -a / source`, sets up
all workspace paths, clears stale caches, and starts the agent for the
competition named in its last line (`--competition`, default
`nomad2018-predict-transparent-conductors`). Accept the rules of the competition
on kaggle.com first ("Join Competition"), otherwise the data download fails.

> ⚠️ `run_linux.sh` must be executed from the repository root directory where
> `config.env` is located — the script looks for it at `./config.env`.

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `docker: permission denied` | User not in `docker` group | `newgrp docker` or restart terminal |
| Docker daemon not starting | WSL2 has no systemd | `sudo service docker start` |
| Slow file access | Files under `/mnt/c/` | Keep project files in `~/` (WSL2 filesystem) |
| `wsl --install` fails | Virtualization disabled | Enable Intel VT-x / AMD-V in BIOS |
| Conda not found after install | `.bashrc` not reloaded | `source ~/.bashrc` |
| `GOOGLE_AI_STUDIO_API_KEY` not set | `secrets.env` not found or not loaded | Run `set -a && source config.env && source secrets.env && set +a` |
| `Cannot connect to the Docker daemon`, `/var/run/docker.sock` missing (Docker Desktop) | WSL integration is off for this distribution, e.g. after a Docker Desktop update | Docker Desktop → Settings → Resources → WSL Integration: enable the distribution, then "Apply & Restart" |
| `chromedriver` or Chrome not found | A competition folder without `description.md` makes the crawler open the Kaggle page | Use MLE-bench data (it brings `description.md`) or install Chrome and chromedriver, see `rdagent/scenarios/kaggle/README.md` |
| `Exec format error` on start | `~/.docker/config.json` contains `"credsStore": "desktop.exe"` from a prior Docker Desktop install | Edit `~/.docker/config.json` and remove the `credsStore` line |
| `config.env` values not loaded | `export $(... xargs)` fails on special characters | Use `set -a && source ./config.env && set +a` instead |
| `rdagent`: `No module named 'rdagent.app'` | The editable install still points to the repository's old path (after moving or renaming the folder) | Run `pip install -e .` again from the repository root |
| `DS_LOCAL_DATA_PATH is not set` | Env var missing when importing modules directly | Run `export DS_LOCAL_DATA_PATH=$(pwd)/workspace && mkdir -p workspace` before any direct `python -c` calls |
