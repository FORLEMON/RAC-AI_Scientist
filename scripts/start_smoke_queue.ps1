param([string]$Batch = 'smoke-20260923')
$ErrorActionPreference = 'Stop'
if ($Batch -notmatch '^[A-Za-z0-9_-]+$') { throw 'Batch must contain only letters, digits, underscores, or hyphens.' }
$repositoryPath = Split-Path -Parent $PSScriptRoot
$batchDirectory = Join-Path $repositoryPath "runs\$Batch"
if (Test-Path -LiteralPath (Join-Path $batchDirectory 'status.json')) {
    throw 'This batch already has execution state. Inspect it before resuming; this launcher never retries existing episodes.'
}
New-Item -ItemType Directory -Path $batchDirectory -Force | Out-Null
$launcher = Start-Process -FilePath 'wsl.exe' -ArgumentList @(
    '-d','Ubuntu','-u','root','--cd',('"' + $repositoryPath + '"'),'--',
    '/usr/bin/python3','-u','scripts/run_smoke_queue.py','--batch',$Batch
) -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $batchDirectory 'launcher.stdout.log') `
    -RedirectStandardError (Join-Path $batchDirectory 'launcher.stderr.log')
$record = @{windows_launcher_pid=$launcher.Id; batch=$Batch; started_at=(Get-Date).ToString('o')}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batchDirectory 'launcher.json') -Encoding utf8
$record | ConvertTo-Json
