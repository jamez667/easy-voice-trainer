# Pester 5 tests for the training watchdog's decision logic.
#
#   Invoke-Pester watchdog.Tests.ps1 -Output Detailed
#
# Step-Watchdog is pure, so every guard is tested without Docker or a
# GPU. Helper functions are tested with mocks / TestDrive fixtures.

BeforeAll {
    . "$PSScriptRoot\watchdog.ps1" -LoadFunctionsOnly

    # Baseline fixtures: healthy state + healthy observation. Tests
    # override only the fields they're about.
    $script:t0 = Get-Date "2026-06-11 09:00:00"
    function New-State([hashtable]$o = @{}) {
        $s = @{ Batch = 4; SpillStrikes = 0; Relaunches = 0; LastCkptTime = $script:t0 }
        foreach ($k in $o.Keys) { $s[$k] = $o[$k] }
        return $s
    }
    function New-Obs([hashtable]$o = @{}) {
        $b = @{
            Running = $true; ExitCode = $null; OOM = $false; SharedMB = 50
            LatestCkptTime = $script:t0; Now = $script:t0.AddMinutes(5)
        }
        foreach ($k in $o.Keys) { $b[$k] = $o[$k] }
        return $b
    }
    $script:cfg = @{ MinBatch = 1; SpillThresholdMB = 1024; StallMinutes = 15; MaxRelaunches = 5 }
}

Describe "Step-Watchdog: shared-memory spill guard" {
    It "first over-threshold sample is a strike, not a kill" {
        $v = Step-Watchdog (New-State) (New-Obs @{ SharedMB = 5000 }) $cfg
        $v.Action | Should -Be "wait"
        $v.State.SpillStrikes | Should -Be 1
        $v.State.Batch | Should -Be 4
    }

    It "second consecutive strike kills and halves batch" {
        $v = Step-Watchdog (New-State @{ SpillStrikes = 1 }) (New-Obs @{ SharedMB = 5000 }) $cfg
        $v.Action | Should -Be "kill-relaunch"
        $v.State.Batch | Should -Be 2
        $v.State.SpillStrikes | Should -Be 0
        $v.State.Relaunches | Should -Be 1
    }

    It "a clean sample between spikes resets the strike count" {
        $v = Step-Watchdog (New-State @{ SpillStrikes = 1 }) (New-Obs @{ SharedMB = 50 }) $cfg
        $v.Action | Should -Be "wait"
        $v.State.SpillStrikes | Should -Be 0
    }

    It "spill at minimum batch is fatal — no relaunch loop" {
        $v = Step-Watchdog (New-State @{ Batch = 1; SpillStrikes = 1 }) (New-Obs @{ SharedMB = 5000 }) $cfg
        $v.Action | Should -Be "fatal"
    }

    It "counter failure (-1) is not treated as a spill" {
        $v = Step-Watchdog (New-State @{ SpillStrikes = 1 }) (New-Obs @{ SharedMB = -1 }) $cfg
        $v.Action | Should -Be "wait"
        $v.State.SpillStrikes | Should -Be 0
    }

    It "last night's failure signature (16 GB shared) trips in exactly two samples" {
        $s = New-State
        $v1 = Step-Watchdog $s (New-Obs @{ SharedMB = 16280 }) $cfg
        $v2 = Step-Watchdog $v1.State (New-Obs @{ SharedMB = 16280 }) $cfg
        $v1.Action | Should -Be "wait"
        $v2.Action | Should -Be "kill-relaunch"
        $v2.State.Batch | Should -Be 2
    }
}

Describe "Step-Watchdog: stall guard" {
    It "new checkpoint advances LastCkptTime and stays healthy" {
        $later = $t0.AddMinutes(5)
        $v = Step-Watchdog (New-State) (New-Obs @{ LatestCkptTime = $later; Now = $later.AddSeconds(30) }) $cfg
        $v.Action | Should -Be "wait"
        $v.State.LastCkptTime | Should -Be $later
    }

    It "no checkpoint within the window is fine" {
        $v = Step-Watchdog (New-State) (New-Obs @{ Now = $t0.AddMinutes(14) }) $cfg
        $v.Action | Should -Be "wait"
    }

    It "no checkpoint past the window kills and relaunches at same batch" {
        $v = Step-Watchdog (New-State) (New-Obs @{ Now = $t0.AddMinutes(16) }) $cfg
        $v.Action | Should -Be "kill-relaunch"
        $v.State.Batch | Should -Be 4
        $v.State.Relaunches | Should -Be 1
    }

    It "stall with OOM in logs also halves batch" {
        $v = Step-Watchdog (New-State) (New-Obs @{ Now = $t0.AddMinutes(16); OOM = $true }) $cfg
        $v.Action | Should -Be "kill-relaunch"
        $v.State.Batch | Should -Be 2
    }

    It "stall at the relaunch cap is fatal" {
        $v = Step-Watchdog (New-State @{ Relaunches = 5 }) (New-Obs @{ Now = $t0.AddMinutes(16) }) $cfg
        $v.Action | Should -Be "fatal"
    }
}

Describe "Step-Watchdog: container exit" {
    It "exit 0 means done" {
        $v = Step-Watchdog (New-State) (New-Obs @{ Running = $false; ExitCode = 0 }) $cfg
        $v.Action | Should -Be "done"
    }

    It "nonzero exit relaunches at same batch" {
        $v = Step-Watchdog (New-State) (New-Obs @{ Running = $false; ExitCode = 137 }) $cfg
        $v.Action | Should -Be "relaunch"
        $v.State.Batch | Should -Be 4
        $v.State.Relaunches | Should -Be 1
    }

    It "OOM exit halves batch" {
        $v = Step-Watchdog (New-State) (New-Obs @{ Running = $false; ExitCode = 1; OOM = $true }) $cfg
        $v.Action | Should -Be "relaunch"
        $v.State.Batch | Should -Be 2
    }

    It "OOM exit at minimum batch relaunches WITHOUT halving below floor" {
        $v = Step-Watchdog (New-State @{ Batch = 1 }) (New-Obs @{ Running = $false; ExitCode = 1; OOM = $true }) $cfg
        $v.Action | Should -Be "relaunch"
        $v.State.Batch | Should -Be 1
    }

    It "crash loop stops at the relaunch cap" {
        $s = New-State
        for ($i = 0; $i -lt 5; $i++) {
            $v = Step-Watchdog $s (New-Obs @{ Running = $false; ExitCode = 1 }) $cfg
            $v.Action | Should -Be "relaunch"
            $s = $v.State
        }
        $v = Step-Watchdog $s (New-Obs @{ Running = $false; ExitCode = 1 }) $cfg
        $v.Action | Should -Be "fatal"
    }

    It "spill then crash: batch reductions accumulate (4 -> 2 -> 1)" {
        $s = (Step-Watchdog (New-State @{ SpillStrikes = 1 }) (New-Obs @{ SharedMB = 5000 }) $cfg).State
        $s.Batch | Should -Be 2
        $v = Step-Watchdog $s (New-Obs @{ Running = $false; ExitCode = 1; OOM = $true }) $cfg
        $v.State.Batch | Should -Be 1
    }
}

Describe "Get-HalvedBatch" {
    It "halves and floors" {
        Get-HalvedBatch 16 1 | Should -Be 8
        Get-HalvedBatch 5 1 | Should -Be 2
    }
    It "clamps at the minimum" {
        Get-HalvedBatch 2 2 | Should -Be 2
        Get-HalvedBatch 1 1 | Should -Be 1
    }
}

Describe "Get-LatestCkpt" {
    It "picks the newest and ignores *.cleaned.ckpt" {
        $d = Join-Path $TestDrive "ckpts"
        New-Item -ItemType Directory $d | Out-Null
        "x" | Set-Content "$d\epoch=7700-step=1.ckpt"
        "x" | Set-Content "$d\epoch=7791-step=2.ckpt"
        "x" | Set-Content "$d\amy_medium.cleaned.ckpt"
        (Get-Item "$d\epoch=7700-step=1.ckpt").LastWriteTime = (Get-Date).AddHours(-2)
        (Get-Item "$d\epoch=7791-step=2.ckpt").LastWriteTime = (Get-Date).AddHours(-1)
        (Get-Item "$d\amy_medium.cleaned.ckpt").LastWriteTime = Get-Date  # newest but excluded

        (Get-LatestCkpt -Dir $d).Name | Should -Be "epoch=7791-step=2.ckpt"
    }

    It "returns nothing for an empty dir" {
        $d = Join-Path $TestDrive "empty"
        New-Item -ItemType Directory $d | Out-Null
        Get-LatestCkpt -Dir $d | Should -BeNullOrEmpty
    }
}

Describe "Get-WslSharedMB" {
    It "sums only vmwp/vmmem instances, ignoring host apps" {
        Mock Get-Counter {
            [pscustomobject]@{ CounterSamples = @(
                [pscustomobject]@{ InstanceName = "pid_1111_luid_x"; CookedValue = 2GB }   # vmwp
                [pscustomobject]@{ InstanceName = "pid_2222_luid_x"; CookedValue = 8GB }   # chrome — ignored
                [pscustomobject]@{ InstanceName = "pid_3333_luid_x"; CookedValue = 0 }     # zero — skipped
            ) }
        }
        Mock Get-Process {
            param($Id)
            switch ([int]($Id | Select-Object -First 1)) {
                1111 { [pscustomobject]@{ ProcessName = "vmwp" } }
                2222 { [pscustomobject]@{ ProcessName = "chrome" } }
            }
        }
        Get-WslSharedMB | Should -Be 2048
    }

    It "returns -1 when the counter read fails" {
        Mock Get-Counter { throw "counter unavailable" }
        Get-WslSharedMB | Should -Be -1
    }
}

Describe "Test-OOMInLogs" {
    It "detects CUDA OOM in container logs" {
        Mock docker { "RuntimeError: CUDA out of memory. Tried to allocate 512 MiB" }
        Test-OOMInLogs | Should -BeTrue
    }
    It "ignores normal training logs" {
        Mock docker { "Epoch 7794/7999  184/201 0:00:49 3.70it/s" }
        Test-OOMInLogs | Should -BeFalse
    }
}
