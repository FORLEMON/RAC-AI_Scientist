param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$Batch = 'smoke-20260923',
    [ValidateRange(2, 3600)][int]$RefreshSeconds = 5,
    [switch]$Once
)
# Read-only dashboard; works in Windows PowerShell 5.1 and PowerShell 7.
$ErrorActionPreference = 'Stop'
$repositoryPath = Split-Path -Parent $PSScriptRoot
$batchPath = Join-Path $repositoryPath "runs\$Batch"
$statusPath = Join-Path $batchPath 'status.json'
$planPath = Join-Path $batchPath 'frozen-plan.json'

function Read-QueueJson([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    # usage.json can briefly be incomplete while the gateway updates it.
    for ($attempt = 0; $attempt -lt 3; $attempt++) {
        try { return (Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json) }
        catch { if ($attempt -eq 2) { return $null }; Start-Sleep -Milliseconds 60 }
    }
}

function Format-Elapsed($Started, $Finished, [double]$Now) {
    if (-not $Started) { return '-' }
    $end = $Now
    if ($Finished) { $end = [double]$Finished }
    $span = [TimeSpan]::FromSeconds([Math]::Max(0, $end - [double]$Started))
    return ('{0:00}:{1:00}:{2:00}' -f [Math]::Floor($span.TotalHours), $span.Minutes, $span.Seconds)
}

do {
    $state = Read-QueueJson $statusPath
    $plan = Read-QueueJson $planPath
    if (-not $Once) { Clear-Host }
    if (-not $state -or -not $plan) {
        Write-Host "Waiting for readable status and frozen plan: $batchPath"
        if ($Once) { exit 1 }
        Start-Sleep -Seconds $RefreshSeconds
        continue
    }

    $recovery = $null; $recoveryPlan = $null; $recoveryPath = $null
    $recoveryRecord = Read-QueueJson (Join-Path $batchPath 'ark-recovery.json')
    if ($recoveryRecord -and $recoveryRecord.batch -match '^[A-Za-z0-9_-]+$') {
        $recoveryPath = Join-Path $repositoryPath ("runs\" + $recoveryRecord.batch)
        $recovery = Read-QueueJson (Join-Path $recoveryPath 'status.json')
        $recoveryPlan = Read-QueueJson (Join-Path $recoveryPath 'frozen-plan.json')
    }
    $now = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
    $ended = 0; $clean = 0; $failed = 0; $scored = 0; $active = 0; $waiting = 0; $cancelled = 0
    $totalCost = 0.0
    $paused = @($state.resources.paused_episodes)
    if ($recovery) { $paused += @($recovery.resources.paused_episodes) }
    $rows = @(foreach ($originalItem in $plan.episodes) {
        $item = $originalItem; $itemState = $state; $itemBatchPath = $batchPath
        if ($recovery -and $recoveryPlan -and $item.host -eq 'ark') {
            $replacement = $recoveryPlan.episodes | Where-Object { $_.planned_episode_id -eq $item.planned_episode_id } | Select-Object -First 1
            if ($replacement) { $item = $replacement; $itemState = $recovery; $itemBatchPath = $recoveryPath }
        }
        $property = $itemState.episodes.PSObject.Properties[$item.episode_id]
        $episode = if ($null -ne $property) { $property.Value } else { $null }
        $phase = if ($episode) { [string]$episode.status } else { 'waiting' }
        switch ($phase) {
            'completed' { $ended++; $clean++ }
            'failed' { $ended++; $failed++ }
            'cancelled' { $cancelled++ }
            'running' { $active++ }
            'scoring' { $active++ }
            default { $waiting++ }
        }
        if ($episode.score.status -eq 'scored') { $scored++ }
        if ($paused -contains $item.episode_id) { $phase = 'paused' }
        $logPath = Join-Path (Join-Path $itemBatchPath 'logs') $item.episode_id
        $usagePath = Join-Path $logPath 'model\usage.json'
        $usage = Read-QueueJson $usagePath
        if (-not $usage -and $episode.model_usage) { $usage = $episode.model_usage }
        $score = '-'
        if ($episode.score.status -eq 'scored' -and $null -ne $episode.score.total_score) {
            $score = '{0:0.000}' -f [double]$episode.score.total_score
        } elseif ($episode.score.status) { $score = [string]$episode.score.status }
        $calls = '-'; $tokens = '-'; $cost = '-'; $last = '-'
        if ($usage) {
            $calls = [string]$usage.calls
            $tokens = '{0:0.00}/{1:0.00}' -f ($usage.input_tokens / 1000000.0), ($usage.output_tokens / 1000000.0)
            $cost = '{0:0.00}' -f [double]$usage.budget_cost_usd
            $totalCost += [double]$usage.budget_cost_usd
        }
        if ($episode.finished_at) {
            $last = [DateTimeOffset]::FromUnixTimeSeconds([long]$episode.finished_at).ToLocalTime().ToString('HH:mm:ss')
        } elseif ($episode) {
            $candidates = @(Get-ChildItem -LiteralPath $logPath -Filter '*.log' -File -ErrorAction SilentlyContinue)
            $modelPath = Join-Path $logPath 'model'
            if (Test-Path -LiteralPath $modelPath) {
                $candidates += @(Get-ChildItem -LiteralPath $modelPath -File -ErrorAction SilentlyContinue)
            }
            $latest = $candidates | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
            if ($latest) { $last = $latest.LastWriteTime.ToString('HH:mm:ss') }
        }
        $hostLabel = switch ($item.host) { 'evo_scientist' {'Evo'} 'ark' {'ARK'} 'agent_laboratory' {'AgentLab'} default {$item.host} }
        $benchmarkLabel = if ($item.benchmark_id -eq 'discoverybench') { 'Discovery' } else { 'CORE' }
        [PSCustomObject]@{
            Host=$hostLabel; Benchmark=$benchmarkLabel; Mode=$item.condition; State=$phase
            Score=$score; Elapsed=(Format-Elapsed $episode.started_at $episode.finished_at $now)
            Calls=$calls; 'Tokens(M) In/Out'=$tokens; 'Cost~USD'=$cost; LastLog=$last
        }
    })
    $overallStatus = $state.status
    if ($recovery -and $recovery.status -eq 'running') { $overallStatus = 'running' }
    Write-Host "RAC | $Batch | $overallStatus | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')" -ForegroundColor Cyan
    Write-Host "Ended $ended/$($plan.episodes.Count) | Cancelled $cancelled | Active $active | Waiting $waiting | Clean $clean | Failed/interrupted $failed | Scored $scored"
    $selection = Read-QueueJson (Join-Path $batchPath 'selection.json')
    if ($selection) { Write-Host "Enabled scope: $($selection.allowed_benchmarks -join ', ') | $($selection.allowed_hosts -join ', ')" -ForegroundColor Cyan }
    $heartbeat = [DateTimeOffset]::Parse($state.updated_at)
    $age = [Math]::Max(0, [Math]::Round(($now - $heartbeat.ToUnixTimeSeconds())))
    $latestResources = $state.resources
    if ($recovery -and $recovery.resources.at -gt $latestResources.at) { $latestResources = $recovery.resources }
    $cpu = if ($latestResources) { '{0:0.0}%' -f $latestResources.cpu_percent } else { '?' }
    $memory = if ($latestResources) { '{0:0.0} GiB' -f ($latestResources.wsl_available_memory_mb / 1024.0) } else { '?' }
    Write-Host ('CPU {0} | WSL memory available {1} | Agent cost estimate ${2:0.00} | Heartbeat {3}s ago' -f $cpu, $memory, $totalCost, $age)
    if ($state.status -eq 'running' -and $age -gt 30) { Write-Host 'WARNING: controller heartbeat is stale; status alone does not prove the queue is alive.' -ForegroundColor Yellow }
    if ($recovery) {
        $recoveryAge = [Math]::Max(0, [Math]::Round($now - [DateTimeOffset]::Parse($recovery.updated_at).ToUnixTimeSeconds()))
        Write-Host "ARK rows show recovery $($recovery.batch): $($recovery.status), admitted slots $($recovery.admitted_parallelism)/2, heartbeat ${recoveryAge}s ago." -ForegroundColor Cyan
        Write-Host "The 6 original zero-call ARK startup failures remain in $batchPath; they are superseded only in this view."
        if ($recovery.status -eq 'running' -and $recoveryAge -gt 30) { Write-Host 'WARNING: ARK recovery heartbeat is stale.' -ForegroundColor Yellow }
    }
    $rows | Format-Table -AutoSize | Out-String -Width 220 | Write-Host
    Write-Host 'Ended includes failures. Score and host exit status are separate; a failed run can still have a valid score.'
    Write-Host 'Calls include attempts; cost is a conservative local estimate and excludes judge fees. LastLog is activity, not percent complete.'
    Write-Host "Events: $batchPath\controller.jsonl"
    Write-Host "Details: $batchPath\logs\<episode_id>\host.stdout.log | host.stderr.log | model\usage.json"
    if ($recovery) { Write-Host "ARK recovery logs and controller.jsonl: $recoveryPath" }
    if (-not $Once) {
        Write-Host "Read-only; refresh every ${RefreshSeconds}s. Ctrl+C closes this view; experiments keep running."
        Start-Sleep -Seconds $RefreshSeconds
    }
} while (-not $Once)
