param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$Batch = 'smoke-20260923-ark-recovery',
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$ParentBatch = 'smoke-20260923'
)
$ErrorActionPreference = 'Stop'
$repositoryPath = Split-Path -Parent $PSScriptRoot
$batchDirectory = Join-Path $repositoryPath "runs\$Batch"
if ((Test-Path -LiteralPath (Join-Path $batchDirectory 'launcher.json')) -or
    (Test-Path -LiteralPath (Join-Path $repositoryPath "runs\$ParentBatch\ark-recovery.json"))) {
    throw 'An ARK recovery is already launched or registered. Inspect existing logs before retrying.'
}
New-Item -ItemType Directory -Path $batchDirectory -Force | Out-Null
$launcher = Start-Process -FilePath 'wsl.exe' -ArgumentList @(
    '-d','Ubuntu','-u','root','--cd',('"' + $repositoryPath + '"'),'--',
    '/usr/bin/python3','-u','scripts/run_ark_recovery.py','--batch',$Batch,'--parent-batch',$ParentBatch
) -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $batchDirectory 'launcher.stdout.log') `
    -RedirectStandardError (Join-Path $batchDirectory 'launcher.stderr.log')
$record = @{windows_launcher_pid=$launcher.Id; batch=$Batch; parent_batch=$ParentBatch; started_at=(Get-Date).ToString('o')}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batchDirectory 'launcher.json') -Encoding utf8
$record | ConvertTo-Json
