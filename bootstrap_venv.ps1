<#
bootstrap_venv.ps1
One-click bootstrap for project virtual environment and dependencies.

Usage:
  Run in PowerShell from project root:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned; .\bootstrap_venv.ps1
  To force recreate the venv: .\bootstrap_venv.ps1 -Force
#>

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

$Force = $false
if ($args -contains '-Force' -or $args -contains '/Force') { $Force = $true }

Write-Host "Project root: $root"

if ((Test-Path ".venv") -and -not $Force) {
    Write-Host ".venv already exists. Use -Force to recreate. Activating..."
} else {
    Write-Host "Creating virtual environment .venv..."
    python -m venv .venv
}

Write-Host "Activating virtual environment..."
& "$root\.venv\Scripts\Activate.ps1"

Write-Host "Upgrading pip, setuptools, wheel..."
python -m pip install --upgrade pip setuptools wheel

if (-not (Test-Path "requirements.txt")) {
    Write-Host "No requirements.txt found — creating a minimal default requirements.txt"
    @"
numpy
pandas
scikit-learn
scipy
matplotlib
seaborn
torch
flwr
opacus
nbclient
nbformat
ipykernel
jupyter
"@ | Out-File -Encoding utf8 requirements.txt
}

Write-Host "Installing Python packages from requirements.txt (may take several minutes)..."
python -m pip install -r requirements.txt

Write-Host "Registering Jupyter kernel (projek-keystroke)..."
python -m ipykernel install --user --name "projek-keystroke" --display-name "projek-keystroke (.venv)" | Out-Null

# Create common output directories
Write-Host "Ensuring output directories exist..."
New-Item -ItemType Directory -Force -Path outputs\reports, outputs\models, outputs\figures, data\processed | Out-Null

Write-Host "Bootstrap complete."
Write-Host "- In VS Code: Select kernel 'projek-keystroke (.venv)'."
Write-Host "- To re-run setup and force recreate venv use: .\\bootstrap_venv.ps1 -Force"

Write-Host "Notes:"
Write-Host "- Installing 'torch' may pick a CPU or CUDA wheel depending on your platform. If you need a specific build, install it manually after this script."
Write-Host "- If installation fails for some packages on Windows + Python 3.14, see project README or open an issue."
