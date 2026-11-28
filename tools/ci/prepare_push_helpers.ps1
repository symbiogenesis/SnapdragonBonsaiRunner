#Requires -Version 7.6
<# Compose one preparation command; the prebuilt Rust maintenance tool owns its lifetime. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Root,
    [Parameter(Mandatory)][string]$SourceRoot,
    [Parameter(Mandatory)][string]$TargetDirectory,
    [Parameter(Mandatory)][string]$ScratchDirectory,
    [Parameter(Mandatory)][long]$BootstrapMilliseconds
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# This is an independently provisioned contributor tool, like Git or Cargo.
# Never select the rejected push helpers as the preparation authority.
$maintenanceTool = $env:BONSAI_TOOLS_PATH
if (-not $maintenanceTool -and $env:LOCALAPPDATA) {
    foreach ($relative in @(
        'SnapdragonBonsai/build/maintenance/aarch64-pc-windows-msvc/release/bonsai-tools.exe',
        'SnapdragonBonsai/build/maintenance/release/bonsai-tools.exe',
        'SwiftBonsai2/build/maintenance/aarch64-pc-windows-msvc/release/bonsai-tools.exe',
        'SwiftBonsai2/build/maintenance/release/bonsai-tools.exe'
    )) {
        $candidate = Join-Path $env:LOCALAPPDATA $relative
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { $maintenanceTool = $candidate; break }
    }
}
if (-not $maintenanceTool) {
    $command = Get-Command bonsai-tools.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) { $maintenanceTool = $command.Source }
}
if (-not $maintenanceTool -or -not [IO.Path]::IsPathFullyQualified($maintenanceTool) -or
    [IO.Path]::GetExtension($maintenanceTool) -ine '.exe' -or
    -not (Test-Path -LiteralPath $maintenanceTool -PathType Leaf)) {
    throw 'Push blocked: automatic preparation requires a checked prebuilt bonsai-tools with release-tool support; select it with BONSAI_TOOLS_PATH or install it in the maintenance cache or PATH.'
}

# Keep thirty seconds for checking and five for owned tree teardown inside the
# original 270-second work bound. Full preparation gets at most three minutes.
$timeoutSeconds = [long][Math]::Min(180, [Math]::Floor((235000 - $BootstrapMilliseconds) / 1000))
if ($timeoutSeconds -lt 1) { throw 'Push blocked: automatic preparation cannot fit in the original push allowance.' }
$words = @(
    $env:BONSAI_PUSH_PWSH_PATH, '-NoLogo', '-NoProfile', '-NonInteractive', '-File',
    (Join-Path $SourceRoot 'tools/ci/check_push_sanity.ps1'), '-Root', $Root,
    '-TargetDirectory', $TargetDirectory, '-Prepare'
) | ForEach-Object {
    if (-not $_ -or $_.Contains('"') -or $_.IndexOfAny([char[]]"`r`n`0") -ge 0) {
        throw 'Push blocked: unsupported automatic preparation argument.'
    }
    '"' + $_.Replace('%', '%%') + '"'
}
# Packaged PowerShell must start normally within CMD's already owned native Job.
$batch = Join-Path $ScratchDirectory 'prepare.cmd'
"@echo off`r`nchcp 65001 >nul`r`nset `"ERRORLEVEL=`"`r`n$($words -join ' ') <nul`r`nexit /b %errorlevel%`r`n" | Set-Content -LiteralPath $batch -Encoding utf8NoBOM
$request = Join-Path $ScratchDirectory 'prepare.json'
@{
    operation = 'invoke'
    filePath = Join-Path $env:SystemRoot 'System32/cmd.exe'
    arguments = @('/d', '/c', $batch)
    timeoutSeconds = $timeoutSeconds
    # Hosted release contexts must not turn this push deadline off.
    buildCommand = $false
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $request -Encoding utf8NoBOM
Write-Host "Automatic preparation has at most $timeoutSeconds seconds within the original push allowance; compiler output follows when it exits."
$result = (& $maintenanceTool release-tool --request $request) | ConvertFrom-Json
if ($LASTEXITCODE) { throw 'Push blocked: native automatic preparation failed; no retry will be attempted.' }
if ($result.stdout) { [Console]::Out.Write($result.stdout) }
if ($result.stderr) { [Console]::Error.Write($result.stderr) }
