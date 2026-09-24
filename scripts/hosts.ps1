param(
    [ValidateSet('status', 'build', 'check', 'shell', 'run')]
    [string]$Action = 'status',
    [ValidateSet('all', 'ark', 'agent_laboratory', 'evo_scientist')]
    [string]$HostName = 'all',
    [string]$Distribution = 'Ubuntu',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$HostArguments
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$services = @{ ark = 'ark'; agent_laboratory = 'agent-laboratory'; evo_scientist = 'evo-scientist' }
$selected = if ($HostName -eq 'all') { @('ark', 'agent_laboratory', 'evo_scientist') } else { @($HostName) }
function Invoke-HostWsl([string[]]$Command) {
    & wsl.exe -d $Distribution -u root --cd $repoRoot -- @Command
    if ($LASTEXITCODE -ne 0) { throw "Host command failed with exit code $LASTEXITCODE" }
}
switch ($Action) {
    'status' {
        Invoke-HostWsl @('docker', 'version', '--format', '{{.Server.Version}}')
        foreach ($item in $selected) {
            Invoke-HostWsl @('docker', 'image', 'inspect', "rac-local/$($services[$item]):validated", '--format', '{{.Id}} {{.Size}}')
        }
    }
    'build' {
        $command = @('docker', 'compose', '-f', 'compose.yaml', '-f', 'docker/compose.local.yaml', 'build')
        foreach ($item in $selected) { $command += $services[$item] }
        Invoke-HostWsl $command
    }
    'check' {
        $command = @('python3', 'scripts/check_local_hosts.py')
        if ($HostName -eq 'all') { $command += '--task-runtime' }
        else { $command += @('--host', $HostName) }
        Invoke-HostWsl $command
    }
    { $_ -in 'shell', 'run' } {
        if ($HostName -eq 'all') { throw 'Choose one -HostName for shell or run.' }
        $command = @('docker', 'compose', '-f', 'compose.yaml', '-f', 'docker/compose.local.yaml', 'run', '--rm')
        if ($Action -eq 'shell') { $command += @('--entrypoint', 'bash') }
        $command += $services[$HostName]
        if ($HostArguments) { $command += $HostArguments }
        Invoke-HostWsl $command
    }
}
