# Piper training watchdog — launches training and babysits it in a loop.
#
#   powershell -File watchdog.ps1 -VoiceName myvoice   # batch=4, to epoch 8000
#   powershell -File watchdog.ps1 -VoiceName myvoice -MaxEpochs 9000
#
# Guards (checked every 30 s):
#   1. SHARED-MEMORY SPILL — the silent killer. WDDM happily pages CUDA
#      into system RAM: util reads 100%, no errors, ~10x slowdown
#      (measured: 42 epochs in a night instead of ~500). Detected via
#      the vmwp/vmmem GPU "Shared Usage" perf counter; two consecutive
#      samples over threshold -> kill, HALVE BATCH SIZE, relaunch.
#      (Batch size is the lever that matters — it sets VRAM pressure.
#      Workers only move CPU-side loading.)
#   2. STALL — no new checkpoint landing on the host for 15 min once
#      training is underway -> kill + relaunch (same batch, unless the
#      logs show OOM, which also halves batch).
#   3. EXIT — container exits 0 (max_epochs reached) -> watchdog done.
#      Nonzero exit -> relaunch (OOM check applies), capped at 5
#      relaunches so a hard failure can't loop all night.
#
# Always resumes from the NEWEST non-cleaned .ckpt in
# <TrainingDir>\checkpoints (never the original base checkpoint — that
# mistake silently resets the epoch counter). Checkpoints save to /training/checkpoints every 5
# epochs, so a kill loses at most ~5 epochs.
#
# Decision logic lives in Step-Watchdog (pure function, unit-tested in
# watchdog.Tests.ps1 — run `Invoke-Pester watchdog.Tests.ps1`).
# The main loop only gathers observations and executes its verdict.
#
# Logs to <TrainingDir>\watchdog.log. Ctrl+C stops the watchdog
# but NOT training; `docker stop <container>` for that.

param(
    [string]$TrainingDir = "$HOME\voice-training",
    [string]$VoiceName = "voice",
    [string]$ContainerName = "${ContainerName}",
    [string]$Image = "piper-train",
    [int]$MaxEpochs = 8000,
    [int]$BatchSize = 4,          # safe default for 10 GB cards
    [int]$MinBatchSize = 1,
    [int]$SharedMemThresholdMB = 1024,
    [int]$StallMinutes = 15,
    [int]$MaxRelaunches = 5,
    [switch]$LoadFunctionsOnly    # dot-source for tests without running
)

$ErrorActionPreference = "Stop"
$tr       = $TrainingDir
$log      = "$tr\watchdog.log"
$ckpt_dir = "$tr\checkpoints"

function Write-Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $log -Value $line -Encoding UTF8
    Write-Host $line
}

function Get-LatestCkpt([string]$Dir = $ckpt_dir) {
    return Get-ChildItem "$Dir\*.ckpt" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notlike "*.cleaned.ckpt" } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
}

# Shared GPU memory held by the WSL2 VM (vmwp/vmmem* hosts Docker's CUDA).
# Host apps (games, browsers) don't run inside the VM, so this isolates
# the training container's spill from everything else on the card.
# Returns -1 on counter failure (caller skips the sample).
function Get-WslSharedMB {
    try {
        $samples = (Get-Counter '\GPU Process Memory(*)\Shared Usage' -ErrorAction Stop).CounterSamples
    } catch { return -1 }
    $total = 0
    foreach ($s in $samples) {
        if ($s.CookedValue -le 0) { continue }
        if ($s.InstanceName -match 'pid_(\d+)') {
            $p = Get-Process -Id $Matches[1] -ErrorAction SilentlyContinue
            if ($p -and $p.ProcessName -match '^vm(wp|mem)') { $total += $s.CookedValue }
        }
    }
    return [math]::Round($total / 1MB)
}

function Test-OOMInLogs {
    $tail = docker logs --tail 50 ${ContainerName} 2>&1 | Out-String
    return ($tail -match "CUDA out of memory|OutOfMemoryError")
}

function Get-HalvedBatch([int]$Batch, [int]$Min) {
    return [math]::Max($Min, [math]::Floor($Batch / 2))
}

# Step-Watchdog: the entire decision policy as a pure function.
#
# $State:  @{ Batch; SpillStrikes; Relaunches; LastCkptTime }
# $Obs:    @{ Running; ExitCode; OOM; SharedMB; LatestCkptTime; Now }
# $Config: @{ MinBatch; SpillThresholdMB; StallMinutes; MaxRelaunches }
#
# Returns @{ Action; Reason; State } where Action is one of:
#   wait | done | fatal | relaunch | kill-relaunch
# (relaunch = container already dead; kill-relaunch = stop it first)
function Step-Watchdog($State, $Obs, $Config) {
    $s = @{
        Batch        = $State.Batch
        SpillStrikes = $State.SpillStrikes
        Relaunches   = $State.Relaunches
        LastCkptTime = $State.LastCkptTime
    }

    if (-not $Obs.Running) {
        if ($Obs.ExitCode -eq 0) {
            return @{ Action = "done"; Reason = "training completed (exit 0)"; State = $s }
        }
        if ($s.Relaunches -ge $Config.MaxRelaunches) {
            return @{ Action = "fatal"; Reason = "exit code $($Obs.ExitCode) and relaunch cap ($($Config.MaxRelaunches)) hit"; State = $s }
        }
        $reason = "container exited (code $($Obs.ExitCode))"
        if ($Obs.OOM -and $s.Batch -gt $Config.MinBatch) {
            $s.Batch = Get-HalvedBatch $s.Batch $Config.MinBatch
            $reason = "died with OOM — halving batch to $($s.Batch)"
        }
        $s.Relaunches++
        $s.SpillStrikes = 0
        return @{ Action = "relaunch"; Reason = $reason; State = $s }
    }

    # Guard 1: shared-memory spill (two strikes — single samples can be
    # transient; -1 means the counter read failed, skip).
    if ($Obs.SharedMB -gt $Config.SpillThresholdMB) {
        $s.SpillStrikes++
        if ($s.SpillStrikes -lt 2) {
            return @{ Action = "wait"; Reason = "shared mem $($Obs.SharedMB) MB (strike $($s.SpillStrikes)/2)"; State = $s }
        }
        if ($s.Batch -le $Config.MinBatch) {
            return @{ Action = "fatal"; Reason = "spilling even at batch=$($Config.MinBatch) — needs a human"; State = $s }
        }
        $s.Batch = Get-HalvedBatch $s.Batch $Config.MinBatch
        $s.SpillStrikes = 0
        $s.Relaunches++
        return @{ Action = "kill-relaunch"; Reason = "shared-memory spill ($($Obs.SharedMB) MB) — halving batch to $($s.Batch)"; State = $s }
    }
    $s.SpillStrikes = 0

    # Guard 2: progress / stall.
    if ($Obs.LatestCkptTime -gt $s.LastCkptTime) {
        $s.LastCkptTime = $Obs.LatestCkptTime
        return @{ Action = "wait"; Reason = "healthy — new checkpoint"; State = $s }
    }
    if (($Obs.Now - $s.LastCkptTime).TotalMinutes -gt $Config.StallMinutes) {
        if ($s.Relaunches -ge $Config.MaxRelaunches) {
            return @{ Action = "fatal"; Reason = "stalled and relaunch cap hit"; State = $s }
        }
        $reason = "no new checkpoint in $($Config.StallMinutes) min"
        if ($Obs.OOM -and $s.Batch -gt $Config.MinBatch) {
            $s.Batch = Get-HalvedBatch $s.Batch $Config.MinBatch
            $reason += " — OOM in logs, halving batch to $($s.Batch)"
        }
        $s.Relaunches++
        return @{ Action = "kill-relaunch"; Reason = $reason; State = $s }
    }
    return @{ Action = "wait"; Reason = "healthy"; State = $s }
}

function Start-Training([int]$batch) {
    $latest = Get-LatestCkpt
    if (-not $latest) { Write-Log "FATAL: no resume checkpoint in $ckpt_dir"; exit 1 }
    Write-Log "launching: batch=$batch, resume from $($latest.Name), max_epochs=$MaxEpochs"
    docker rm -f ${ContainerName} 2>&1 | Out-Null
    docker run -d --name ${ContainerName} --gpus all --shm-size=8g `
      -e PYTHONUNBUFFERED=1 `
      -v "${tr}:/training" `
      $Image fit `
        --data.voice_name $VoiceName `
        --data.csv_path /training/piper_dataset/metadata.csv `
        --data.audio_dir /training/piper_dataset/wavs `
        --model.sample_rate 22050 `
        --data.espeak_voice en-us `
        --data.cache_dir /training/piper_cache `
        --data.config_path /training/$VoiceName.onnx.json `
        --data.batch_size $batch `
        --data.num_workers 2 `
        --trainer.accelerator gpu `
        --trainer.devices 1 `
        --trainer.precision 32 `
        --trainer.max_epochs $MaxEpochs `
        --trainer.log_every_n_steps 10 `
        --trainer.callbacks+="lightning.pytorch.callbacks.ModelCheckpoint" `
        --trainer.callbacks.every_n_epochs=5 `
        --trainer.callbacks.save_top_k=-1 `
        --trainer.callbacks.dirpath="/training/checkpoints" `
        --ckpt_path "/training/checkpoints/$($latest.Name)" 2>&1 | Out-Null
}

function Stop-Training($reason) {
    Write-Log "killing container: $reason"
    docker stop -t 5 ${ContainerName} 2>&1 | Out-Null
}

if ($LoadFunctionsOnly) { return }

# --- main loop: observe, decide (Step-Watchdog), execute ---
$state = @{
    Batch        = $BatchSize
    SpillStrikes = 0
    Relaunches   = 0
    LastCkptTime = (Get-LatestCkpt).LastWriteTime
}
$config = @{
    MinBatch         = $MinBatchSize
    SpillThresholdMB = $SharedMemThresholdMB
    StallMinutes     = $StallMinutes
    MaxRelaunches    = $MaxRelaunches
}
Start-Training $state.Batch

while ($true) {
    Start-Sleep -Seconds 30

    $running = (docker inspect -f '{{.State.Running}}' ${ContainerName} 2>$null) -eq "true"
    $obs = @{
        Running        = $running
        ExitCode       = if ($running) { $null } else { [int](docker inspect -f '{{.State.ExitCode}}' ${ContainerName} 2>$null) }
        OOM            = Test-OOMInLogs
        SharedMB       = Get-WslSharedMB
        LatestCkptTime = (Get-LatestCkpt).LastWriteTime
        Now            = Get-Date
    }

    $verdict = Step-Watchdog $state $obs $config
    $state = $verdict.State
    Write-Log "$($verdict.Action): $($verdict.Reason) [batch=$($state.Batch) relaunches=$($state.Relaunches) shared=$($obs.SharedMB)MB]"

    switch ($verdict.Action) {
        "done"          { exit 0 }
        "fatal"         { Stop-Training $verdict.Reason; exit 1 }
        "kill-relaunch" { Stop-Training $verdict.Reason; Start-Training $state.Batch; $state.LastCkptTime = (Get-LatestCkpt).LastWriteTime }
        "relaunch"      { Start-Training $state.Batch; $state.LastCkptTime = (Get-LatestCkpt).LastWriteTime }
        "wait"          { }
    }
}
