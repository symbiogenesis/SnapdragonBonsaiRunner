#Requires -Version 7.6
<# Parse selected scripts without executing them or applying style rules. #>
[CmdletBinding()]
param([Parameter(Mandatory)][string]$Root, [Parameter(Mandatory)][string]$PathsFile)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$failed = $false
foreach ($path in (Get-Content -LiteralPath $PathsFile -Raw | ConvertFrom-Json)) {
    $tokens = $null; $parseErrors = $null
    [void][Management.Automation.Language.Parser]::ParseFile((Join-Path $Root $path), [ref]$tokens, [ref]$parseErrors)
    foreach ($problem in $parseErrors) {
        [Console]::Error.WriteLine("${path}:$($problem.Extent.StartLineNumber): $($problem.Message)")
        $failed = $true
    }
}
if ($failed) { exit 1 }
'Selected PowerShell syntax passed.'
