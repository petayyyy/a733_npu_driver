param(
    [string[]]$Models = @("135m", "360m", "1p7b"),
    [int[]]$Windows = @(32, 64, 128, 256),
    [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"

$GitBash = "C:\Apps\System\Git\bin\bash.exe"
if (-not (Test-Path $GitBash)) {
    throw "Git Bash not found: $GitBash"
}

$Repo = (Get-Location).Path
$Mount = "${Repo}:/workspace"
$Image = "ubuntu-npu:v2.0.10.1"
$DockerArgs = @("run", "--rm", "--cpus", "10", "--memory", "24g", "-v", $Mount, "-w", "/workspace", $Image)
$env:DOCKER_RUN_ARGS = "--cpus 10 --memory 24g"

$ModelInfo = @{
    "135m" = @{ Slug = "smollm2_135m" }
    "360m" = @{ Slug = "smollm2_360m" }
    "1p7b" = @{ Slug = "smollm2_1p7b" }
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
        $Base = "b1_$($Info.Slug)_w$Window"
        $Name = "${Base}_int16"
        $OutDir = "work/generated/$Base"
        $PackageDir = "work/model-packages/$Name/int16"
        $JsonName = "b1-$($Info.Slug.Replace('_', '-'))-w$Window-int16-host-vs-fp.json"
        $Json = "logs/host/$JsonName"
        $ConvertLog = "logs/host/b1-$($Info.Slug.Replace('_', '-'))-w$Window-int16-convert.log"
        $ConvertErr = "logs/host/b1-$($Info.Slug.Replace('_', '-'))-w$Window-int16-convert.err.log"

        if ($SkipExisting -and (Test-Path $Json)) {
            Write-Host "skip existing $Model W=$Window"
            continue
        }

        if (-not (Test-Path "$PackageDir/host_output_0.txt")) {
            Write-Host "== $Model W=${Window}: ACUITY int16 convert/export =="
            $PreviousErrorActionPreference = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            & $GitBash scripts/host/convert_onnx_to_nbg.sh `
                --name $Name `
                --onnx "$OutDir/real_llm.onnx" `
                --dataset "$OutDir/dataset.txt" `
                --quant int16 `
                --inputs token_ids `
                --input-size-list "$Window" `
                --outputs logits `
                > $ConvertLog 2> $ConvertErr
            $ConvertExit = $LASTEXITCODE
            $ErrorActionPreference = $PreviousErrorActionPreference
            if ($ConvertExit -ne 0) {
                Write-Host "conversion wrapper exit code: $ConvertExit"
            }
            if (-not (Test-Path "$PackageDir/host_output_0.txt")) {
                throw "missing host output after conversion: $PackageDir/host_output_0.txt"
            }
        }

        Write-Host "== $Model W=${Window}: ACUITY host vs FP =="
        Invoke-CheckedDocker ($DockerArgs + @(
            "python3", "scripts/host/compare_acuity_host_to_oracle.py",
            "--package-dir", $PackageDir,
            "--oracle", "work/generated/${Base}_oracle/fp_oracle.npz",
            "--model-info", "$OutDir/model_info.json",
            "--output-json", $Json
        ))
    }
}
