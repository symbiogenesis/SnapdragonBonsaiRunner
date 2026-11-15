#Requires -Version 7.6
<# Compose the native checker's selected policy calls in one PowerShell process. #>
[CmdletBinding()]
param([Parameter(Mandatory)][string]$CallsFile)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
foreach ($call in (Get-Content -LiteralPath $CallsFile -Raw | ConvertFrom-Json -AsHashtable)) {
    # Selection and named arguments are serialized by the trusted native checker.
    # Invoke in a child scope so one script's functions/preferences cannot replace
    # the next script's policy; every path remains in committed checking policy.
    $parameters = $call.parameters
    $detail = if ($parameters -and $parameters.Count -gt 0) { ' (' + ($parameters | ConvertTo-Json -Compress) + ')' } else { '' }
    $global:LASTEXITCODE = 0
    & (Join-Path $PSScriptRoot $call.script) @parameters
    if ($LASTEXITCODE) { throw "Push blocked: $($call.script)$detail exited $LASTEXITCODE." }
}
