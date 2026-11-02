[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $Plan,
    [Parameter(Mandatory = $true)] [string] $CheckpointRoot,
    [string[]] $ValidationManifest = @(),
    [string[]] $AnchorManifest = @(),
    [string] $ReplayRoot = "",
    [ValidateRange(1, 100000)] [int] $ReplayWindowsPerChunk = 64,
    [string] $InitialCheckpoint = "",
    [string] $StagingRoot = "",
    [string] $PythonExe = "",
    [ValidateRange(1, 10000)] [int] $Epochs = 1,
    [ValidateRange(0, 100000)] [int] $CheckpointEverySteps = 25,
    [ValidateSet("auto", "cpu", "cuda")] [string] $Device = "auto",
    [ValidateSet("shuffle", "plan")] [string] $ChunkOrder = "shuffle",
    [int] $ShuffleSeed = 20251022,
    [switch] $NewWindow
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = if ($PythonExe) {
    [System.IO.Path]::GetFullPath($PythonExe)
} else {
    Join-Path $ProjectRoot "..\.phase0_env\Scripts\python.exe"
}
$Runner = Join-Path $PSScriptRoot "phase6_chunk_runner.py"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found at $Python. Pass -PythonExe or create .phase0_env first."
}

$RunnerArgs = @(
    "--plan", (Resolve-Path -LiteralPath $Plan).Path,
    "--checkpoint-root", [System.IO.Path]::GetFullPath($CheckpointRoot),
    "--epochs", $Epochs.ToString(),
    "--checkpoint-every-steps", $CheckpointEverySteps.ToString(),
    "--replay-windows-per-chunk", $ReplayWindowsPerChunk.ToString(),
    "--device", $Device,
    "--chunk-order", $ChunkOrder,
    "--shuffle-seed", $ShuffleSeed.ToString()
)
if ($InitialCheckpoint) {
    $RunnerArgs += @("--initial-checkpoint", [System.IO.Path]::GetFullPath($InitialCheckpoint))
}
if ($ReplayRoot) {
    $RunnerArgs += @("--replay-root", [System.IO.Path]::GetFullPath($ReplayRoot))
}
foreach ($ManifestPath in $ValidationManifest) {
    $RunnerArgs += @("--validation-manifest", [System.IO.Path]::GetFullPath($ManifestPath))
}
foreach ($ManifestPath in $AnchorManifest) {
    $RunnerArgs += @("--anchor-manifest", [System.IO.Path]::GetFullPath($ManifestPath))
}
if ($StagingRoot) {
    $RunnerArgs += @("--staging-root", [System.IO.Path]::GetFullPath($StagingRoot))
}

if ($NewWindow) {
    function Quote-PowerShellLiteral([string] $Value) {
        return "'" + $Value.Replace("'", "''") + "'"
    }
    $argumentText = ($RunnerArgs | ForEach-Object { Quote-PowerShellLiteral ([string] $_) }) -join " "
    $commandText = "& $(Quote-PowerShellLiteral $Python) -u $(Quote-PowerShellLiteral $Runner) $argumentText`n"
    $commandText += "`$code = `$LASTEXITCODE`nWrite-Host ''`nWrite-Host ('Runner exited with code ' + `$code)`nRead-Host 'Press Enter to close this window'`nexit `$code"
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($commandText))
    Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoExit", "-EncodedCommand", $encoded) -WorkingDirectory $ProjectRoot -WindowStyle Normal
    Write-Host "Opened a visible PowerShell window. The runner will require you to type TRAIN before each chunk."
    return
}

Push-Location $ProjectRoot
$RunnerExitCode = 0
try {
    & $Python -u $Runner @RunnerArgs
    $RunnerExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
return $RunnerExitCode
