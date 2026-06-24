param(
    [string[]]$Models = @("135m", "360m", "1p7b"),
    [int[]]$Windows = @(32, 64, 128, 256),
    [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"

$Repo = (Get-Location).Path
$Mount = "${Repo}:/workspace"
$Image = "ubuntu-npu:v2.0.10.1"
$DockerArgs = @("run", "--rm", "--cpus", "10", "--memory", "24g", "-v", $Mount, "-w", "/workspace", $Image)

$ModelInfo = @{
    "135m" = @{ Slug = "smollm2_135m"; Dir = "work/models/smollm2-135m-instruct" }
    "360m" = @{ Slug = "smollm2_360m"; Dir = "work/models/smollm2-360m-instruct" }
    "1p7b" = @{ Slug = "smollm2_1p7b"; Dir = "work/models/smollm2-1.7b-instruct" }
}

$PromptTokens = @(
    1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308,
    34519, 28, 7018, 411, 407, 19712, 8182, 2, 198, 1, 4093, 198,
    504, 3575, 282, 4649, 314, 2, 198, 1, 520, 9531, 198
)

function Get-WindowTokens([int]$Window) {
    $count = [Math]::Min($Window, $PromptTokens.Count)
    $start = $PromptTokens.Count - $count
    return (($PromptTokens[$start..($PromptTokens.Count - 1)]) -join ",")
}

function Invoke-CheckedDocker([string[]]$DockerCliArgs) {
    & docker @DockerCliArgs
    if ($LASTEXITCODE -ne 0) {
        throw "docker failed with exit code ${LASTEXITCODE}: docker $($DockerCliArgs -join ' ')"
    }
}

foreach ($Model in $Models) {
    if (-not $ModelInfo.ContainsKey($Model)) {
        throw "unknown model key: $Model"
    }
    $Info = $ModelInfo[$Model]
    foreach ($Window in $Windows) {
        $OutDir = "work/generated/b1_$($Info.Slug)_w$Window"
        $Oracle = "work/generated/b1_$($Info.Slug)_w${Window}_oracle/fp_oracle.npz"
        $Json = "logs/host/b1-$($Info.Slug.Replace('_', '-'))-w$Window-onnxruntime-vs-fp.json"

        if ($SkipExisting -and (Test-Path $Json)) {
            Write-Host "skip existing $Model W=$Window"
            continue
        }

        Write-Host "== $Model W=${Window}: build ONNX =="
        Invoke-CheckedDocker ($DockerArgs + @(
            "python3", "scripts/host/make_real_llm_onnx.py",
            "--model-dir", $Info.Dir,
            "--output-dir", $OutDir,
            "--seq-len", "$Window",
            "--tokens", (Get-WindowTokens $Window),
            "--no-check"
        ))

        Write-Host "== $Model W=${Window}: FP oracle =="
        Invoke-CheckedDocker ($DockerArgs + @(
            "python3", "scripts/host/dump_real_llm_oracle.py",
            "--model-dir", $Info.Dir,
            "--tokens", "$OutDir/token_ids.npy",
            "--output", $Oracle,
            "--seq-len", "$Window"
        ))

        Write-Host "== $Model W=${Window}: ONNX Runtime vs FP =="
        Invoke-CheckedDocker ($DockerArgs + @(
            "python3", "scripts/host/compare_onnxruntime_to_oracle.py",
            "--onnx", "$OutDir/real_llm.onnx",
            "--tokens", "$OutDir/token_ids.npy",
            "--oracle", $Oracle,
            "--model-info", "$OutDir/model_info.json",
            "--output-json", $Json,
            "--threads", "10"
        ))
    }
}
