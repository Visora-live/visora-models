$ErrorActionPreference = "Stop"

# Activate venv (stored outside OneDrive to avoid application control blocks)
$venvActivate = "C:\visora_venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
} else {
    Write-Error "venv not found at C:\visora_venv. Run setup again."
    exit 1
}

# Load .env into environment variables
if (Test-Path "$PSScriptRoot\.env") {
    Get-Content "$PSScriptRoot\.env" | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), 'Process')
        }
    }
}

python "$PSScriptRoot\run_weapon_webcam.py"
