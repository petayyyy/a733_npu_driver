param(
    [int]$start = 2,
    [int]$end = 23
)

$repoRoot = $PSScriptRoot + "\..\.."
$repoRoot = (Resolve-Path $repoRoot).Path
$modelsDir = "$repoRoot\work\ai-sdk\ZIFENG278-ai-sdk\models"
$sdkScripts = "$repoRoot\work\ai-sdk\ZIFENG278-ai-sdk\scripts"
$dockerRepoRoot = "C:/Users/ilyah/Documents/Work/a733_npu_driver"
$DOCKER_IMAGE = "ubuntu-npu:v2.0.10.1"
$TARGET = "VIP9000NANODI_PLUS_PID0X1000003B"

for ($block = $start; $block -le $end; $block++) {
    $name = "qwen25_05b_w32_block$block"
    $genDir = "$repoRoot\work\generated\qwen25_05b_w32_block$block"
    $modelDir = "$modelsDir\$name"

    Write-Host "=== Compiling $name ===" -ForegroundColor Cyan

    # Clean and recreate model dir
    if (Test-Path $modelDir) { Remove-Item -Recurse -Force $modelDir }
    New-Item -ItemType Directory -Path $modelDir -Force | Out-Null

    # Copy ONNX
    Copy-Item "$genDir\real_llm.onnx" "$modelDir\$name.onnx"

    # Copy dataset + payload
    Copy-Item "$genDir\dataset.txt" "$modelDir\dataset.txt"
    Copy-Item "$genDir\hidden_in.npy" "$modelDir\hidden_in.npy"

    # Copy inputs_outputs.txt with LF line endings
    $io = Get-Content "$genDir\inputs_outputs.txt" -Raw
    if (-not $io) {
        $io = "--inputs hidden_in --input-size-list 32,896 --outputs layer${block}_mlp_resid"
    }
    [System.IO.File]::WriteAllText("$modelDir\inputs_outputs.txt", $io.TrimEnd() + "`n", [System.Text.Encoding]::ASCII)

    # Copy pegasus scripts
    Copy-Item "$sdkScripts\pegasus_import.sh" "$modelDir\"
    Copy-Item "$sdkScripts\pegasus_quantize.sh" "$modelDir\"
    Copy-Item "$sdkScripts\pegasus_inference.sh" "$modelDir\"
    Copy-Item "$sdkScripts\pegasus_export_ovx_nbg.sh" "$modelDir\"
    Copy-Item "$sdkScripts\pegasus_setup.sh" "$modelDir\env.sh"

    # Run Docker conversion
    $containerCmd = @"
set -euo pipefail
export ACUITY_PATH=/root/acuity-toolkit-whl-6.30.22/bin
export VIV_SDK=/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools
source env.sh v3
bash pegasus_import.sh $name
python3 -c "
from pathlib import Path
p = Path('${name}/${name}_inputmeta.yml')
if p.exists():
    t = p.read_text()
    t = t.replace('category: image', 'category: undefined')
    t = t.replace('reverse_channel: true', 'reverse_channel: false')
    p.write_text(t)
"
bash pegasus_quantize.sh $name int16
bash pegasus_inference.sh $name int16
bash pegasus_export_ovx_nbg.sh $name int16 $TARGET /root/Vivante_IDE/VivanteIDE5.11.0/cmdtools
"@

    $logFile = "$repoRoot\logs\host\q2-gate2c-block${block}-int16-convert.log"
    Write-Host "  Docker running..." -ForegroundColor Yellow

    $result = & docker run --rm --cpus 10 --memory 24g `
        -v "${dockerRepoRoot}:/workspace" `
        -w "/workspace/work/ai-sdk/ZIFENG278-ai-sdk/models" `
        $DOCKER_IMAGE `
        bash -lc $containerCmd 2>&1

    $result | Out-File -FilePath $logFile -Encoding utf8

    # Check for success
    $nbPath = "$modelDir\wksp\${name}_int16_nbg_unify\network_binary.nb"
    if (Test-Path $nbPath) {
        $size = (Get-Item $nbPath).Length
        Write-Host "  OK: NBG $size bytes" -ForegroundColor Green
    } else {
        Write-Host "  FAILED: no NBG produced" -ForegroundColor Red
        $result | Select-Object -Last 20
        break
    }
}

Write-Host "=== Batch done ===" -ForegroundColor Cyan
