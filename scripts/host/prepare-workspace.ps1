param(
    [string]$Root = (Get-Location).Path,
    [string]$AcuityImage = "ubuntu-npu:v2.0.10.1",
    [switch]$Strict
)

$ErrorActionPreference = "Stop"

function Write-Status {
    param([string]$Message)
    Write-Host "[a733] $Message"
}

function Ensure-Dir {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Invoke-NativeSoft {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    $OldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $Output = & $Command 2>&1
        $ExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $OldPreference
    }

    [pscustomobject]@{
        ExitCode = $ExitCode
        Output = $Output
    }
}

$Root = (Resolve-Path -LiteralPath $Root).Path
$Dirs = @(
    "logs/host",
    "logs/board",
    "models/onnx",
    "models/nbg",
    "models/calibration",
    "work/acuity",
    "work/ai-sdk"
)

Write-Status "workspace root: $Root"
foreach ($dir in $Dirs) {
    Ensure-Dir -Path (Join-Path $Root $dir)
}

if (-not $env:DOCKER_CONFIG) {
    $env:DOCKER_CONFIG = Join-Path $Root "work/docker-config"
    Ensure-Dir -Path $env:DOCKER_CONFIG
}

$ReportPath = Join-Path $Root "logs/host/prepare-workspace.txt"
"A733 host workspace preparation" | Set-Content -LiteralPath $ReportPath
"Root=$Root" | Add-Content -LiteralPath $ReportPath
"AcuityImage=$AcuityImage" | Add-Content -LiteralPath $ReportPath
"DockerConfig=$env:DOCKER_CONFIG" | Add-Content -LiteralPath $ReportPath

$Docker = Get-Command docker -ErrorAction SilentlyContinue
if ($Docker) {
    Write-Status "docker found: $($Docker.Source)"
    "Docker=$($Docker.Source)" | Add-Content -LiteralPath $ReportPath
    $DockerVersion = Invoke-NativeSoft { docker --version }
    $DockerVersion.Output | Add-Content -LiteralPath $ReportPath

    $Inspect = Invoke-NativeSoft { docker image inspect $AcuityImage }
    $InspectExit = $Inspect.ExitCode
    if ($InspectExit -eq 0) {
        Write-Status "ACUITY image available: $AcuityImage"
        "AcuityImagePresent=true" | Add-Content -LiteralPath $ReportPath
    } else {
        Write-Status "ACUITY image not present locally: $AcuityImage"
        "AcuityImagePresent=false" | Add-Content -LiteralPath $ReportPath
        "DockerInspectExit=$InspectExit" | Add-Content -LiteralPath $ReportPath
        $Inspect.Output | Add-Content -LiteralPath $ReportPath
        if ($Strict) {
            throw "Missing required ACUITY Docker image: $AcuityImage"
        }
    }
} else {
    Write-Status "docker not found"
    "Docker=missing" | Add-Content -LiteralPath $ReportPath
    if ($Strict) {
        throw "Docker is required for ACUITY conversion work"
    }
}

Write-Status "prepared directories and wrote $ReportPath"
