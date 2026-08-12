param(
    [Parameter(Mandatory = $true)]
    [string]$Archive,

    [string]$Destination = "outputs/models/dnabert2_kaggle_best"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$archivePath = (Resolve-Path $Archive).Path
$destinationPath = Join-Path $repoRoot $Destination

if (Test-Path -LiteralPath $destinationPath) {
    Remove-Item -LiteralPath $destinationPath -Recurse -Force
}
New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null

$temporaryDirectory = Join-Path ([System.IO.Path]::GetTempPath()) ("seqtrainer-dnabert2-" + [guid]::NewGuid())
try {
    Expand-Archive -LiteralPath $archivePath -DestinationPath $temporaryDirectory -Force
    $bundleRoot = Get-ChildItem -LiteralPath $temporaryDirectory -Directory | Select-Object -First 1
    if ($null -eq $bundleRoot) {
        throw "The archive does not contain a model bundle directory."
    }

    foreach ($file in @("manifest.json", "metrics.csv", "metrics.json", "history.csv", "input_split_audit.json", "config.json")) {
        Copy-Item -LiteralPath (Join-Path $bundleRoot.FullName $file) -Destination $destinationPath
    }
    $checkpoints = Join-Path $destinationPath "checkpoints"
    New-Item -ItemType Directory -Path $checkpoints -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $bundleRoot.FullName "checkpoints/best_model.pt") -Destination $checkpoints
}
finally {
    if (Test-Path -LiteralPath $temporaryDirectory) {
        Remove-Item -LiteralPath $temporaryDirectory -Recurse -Force
    }
}

$manifestPath = Join-Path $destinationPath "manifest.json"
$checkpointPath = Join-Path $destinationPath "checkpoints/best_model.pt"
if (-not (Test-Path -LiteralPath $manifestPath) -or -not (Test-Path -LiteralPath $checkpointPath)) {
    throw "The prepared bundle is incomplete. Expected manifest.json and checkpoints/best_model.pt."
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.model.family -ne "dnabert2") {
    throw "Expected a DNABERT2 bundle, found model family '$($manifest.model.family)'."
}
if ($manifest.model.params.mode -ne "full_finetune") {
    throw "Expected a full-fine-tuning checkpoint, found mode '$($manifest.model.params.mode)'."
}

Write-Output "Prepared Kaggle DNABERT2 annotation bundle: $destinationPath"
Write-Output "Checkpoint: $checkpointPath"
Write-Output "Manifest:   $manifestPath"
Write-Output "Threshold:  $($manifest.evaluation.selected_threshold) (validation MCC)"
