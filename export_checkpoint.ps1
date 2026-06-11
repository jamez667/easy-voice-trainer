# Export a Lightning checkpoint to ONNX using the same training image.
#
# Usage:
#   .\export_checkpoint.ps1 -VoiceName myvoice                       # latest ckpt
#   .\export_checkpoint.ps1 -VoiceName myvoice -CkptName "epoch=8000-step=X.ckpt"
#
# Writes <TrainingDir>\exports\<ckpt-name>.onnx (+ the voice's
# .onnx.json copied alongside — Piper inference needs the pair).
param(
    [string]$TrainingDir = "$HOME\.easy-voice-trainer",
    [string]$VoiceName = "voice",
    [string]$Image = "piper-train",
    [string]$CkptName = ""
)
$ErrorActionPreference = "Stop"
$tr = $TrainingDir
$ckpts = "$tr\checkpoints"

if ($CkptName) {
    $ckpt = Join-Path $ckpts $CkptName
} else {
    $ckpt = Get-ChildItem -Path $ckpts -Filter "*.ckpt" |
        Where-Object { $_.Name -notlike "*.cleaned.ckpt" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $ckpt -or -not (Test-Path $ckpt)) {
    Write-Error "no checkpoint found in $ckpts"
    exit 1
}
Write-Host "checkpoint: $ckpt"

$name = [System.IO.Path]::GetFileNameWithoutExtension($ckpt)
New-Item -ItemType Directory -Path "$tr\exports" -Force | Out-Null

docker run --rm -v "${tr}:/training" --entrypoint python $Image `
  -m piper.train.export_onnx `
  --checkpoint "/training/checkpoints/$(Split-Path $ckpt -Leaf)" `
  --output-file "/training/exports/$name.onnx"

if (-not (Test-Path "$tr\exports\$name.onnx")) {
    Write-Error "export failed — no .onnx written"
    exit 1
}
$cfg = "$tr\$VoiceName.onnx.json"
if (Test-Path $cfg) {
    Copy-Item $cfg "$tr\exports\$name.onnx.json" -Force
} else {
    Write-Warning "no $cfg found — piper inference needs the .onnx.json next to the .onnx"
}
$mb = [math]::Round((Get-Item "$tr\exports\$name.onnx").Length / 1MB, 1)
Write-Host "exported: $tr\exports\$name.onnx ($mb MB)"
