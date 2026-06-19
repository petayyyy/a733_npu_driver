param(
    [Parameter(Mandatory = $true)]
    [string]$Host,

    [string]$User,
    [int]$Port = 22,
    [string]$RemoteDir = "a733_npu_driver",
    [string]$VpmRunArgs = "",
    [switch]$BatchMode
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host "[a733] $FilePath $($Arguments -join ' ')"
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath exited with code $LASTEXITCODE"
    }
}

function Escape-ShSingleQuoted {
    param([string]$Value)
    return $Value.Replace("'", "'\''")
}

$Target = if ($User) { "$User@$Host" } else { $Host }
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BoardDir = Resolve-Path -LiteralPath (Join-Path $ScriptRoot "..\board")
$BoardScripts = Get-ChildItem -LiteralPath $BoardDir -Filter "*.sh" |
    ForEach-Object { $_.FullName }

if ($BoardScripts.Count -eq 0) {
    throw "No board scripts found in $BoardDir"
}

$SshOptions = @(
    "-p", "$Port",
    "-o", "StrictHostKeyChecking=accept-new"
)
if ($BatchMode) {
    $SshOptions += @("-o", "BatchMode=yes")
}

$ScpOptions = @(
    "-P", "$Port",
    "-o", "StrictHostKeyChecking=accept-new"
)
if ($BatchMode) {
    $ScpOptions += @("-o", "BatchMode=yes")
}

$RemoteBoardDir = "$RemoteDir/scripts/board"
$EscapedRemoteBoardDir = Escape-ShSingleQuoted $RemoteBoardDir
$EscapedRemoteDir = Escape-ShSingleQuoted $RemoteDir

Invoke-Checked ssh ($SshOptions + @($Target, "mkdir -p '$EscapedRemoteBoardDir'"))
Invoke-Checked scp ($ScpOptions + $BoardScripts + @("${Target}:$RemoteBoardDir/"))

$RemoteCommand = "cd '$EscapedRemoteDir' && chmod +x scripts/board/*.sh"
if ($VpmRunArgs) {
    $EscapedArgs = Escape-ShSingleQuoted $VpmRunArgs
    $RemoteCommand += " && A733_VPM_RUN_ARGS='$EscapedArgs' scripts/board/a733-g0-g1-smoke.sh"
} else {
    $RemoteCommand += " && scripts/board/a733-g0-g1-smoke.sh"
}

Invoke-Checked ssh ($SshOptions + @($Target, $RemoteCommand))
