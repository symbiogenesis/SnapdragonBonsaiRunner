#Requires -Version 7.6
<#
.SYNOPSIS
Launches native chat with the Rust inference engine development default.
.DESCRIPTION
Uses a prebuilt bonsai-tools executable. Rust owns startup and application policy.
.EXAMPLE
./scripts/Start-Chat.ps1
.EXAMPLE
./scripts/Start-Chat.ps1 -PublisherEngine
.EXAMPLE
./scripts/Start-Chat.ps1 -GraphicsBackend dx12
#>
[CmdletBinding()]
[OutputType([string])]
param(
    [switch]$NativeEngine,
    [switch]$PublisherEngine,
    [switch]$RequireBrokerPipe,
    [ValidateSet(1, 4, 8, 12)]
    [int]$NativeRowWorkers,
    [ValidateSet('disabled', 'latest', 'prompt-end-and-latest')]
    [string]$PromptReuse,
    [ValidateRange(1, 8)]
    [int]$PromptReuseCheckpoints,
    [ValidateRange(1, 9223372036854775807)]
    [long]$PromptReuseCheckpointBytes,
    [ValidateSet('f16', 'rsq4', 'rsq4k3v')]
    [string]$KvCache,
    [ValidateSet(8, 16, 32)]
    [int]$CpuPrefillTokens,
    [ValidateSet('sequential', 'wy-f32')]
    [string]$CpuPrefillGdn,
    [ValidateSet('none', 'suffix')]
    [string]$SpecType,
    [ValidateRange(1, 8)]
    [int]$DraftMax,
    [ValidateSet('x1', 'x2')]
    [string]$CpuFamily,
    [ValidateSet('auto', 'cpu')]
    [string]$Placement,
    [switch]$UseRepackedPtq1,
    [switch]$UseOriginalPq2,
    [ValidateSet('dx12', 'vulkan', 'gl')]
    [string]$GraphicsBackend,
    [string]$DataDirectory,
    [string]$ToolPath = $env:BONSAI_TOOLS_PATH
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

if (-not $ToolPath -and $env:LOCALAPPDATA) {
    foreach ($relative in @(
        'SwiftBonsai2/build/maintenance/aarch64-pc-windows-msvc/release/bonsai-tools.exe',
        'SwiftBonsai2/build/maintenance/release/bonsai-tools.exe'
    )) {
        $cachedTool = Join-Path $env:LOCALAPPDATA $relative
        if (Test-Path -LiteralPath $cachedTool -PathType Leaf) {
            $ToolPath = $cachedTool
            break
        }
    }
}
if (-not $ToolPath) {
    $command = Get-Command bonsai-tools.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) { $ToolPath = $command.Source }
}
if (-not $ToolPath -or -not (Test-Path -LiteralPath $ToolPath -PathType Leaf)) {
    $message = 'Select a prebuilt bonsai-tools executable with -ToolPath or BONSAI_TOOLS_PATH.'
    if ($ToolPath) {
        $message += ' Correct the selected path, or remove the override to use default discovery.'
    }
    if ($env:LOCALAPPDATA) {
        $manifest = (Join-Path (Split-Path $PSScriptRoot -Parent) 'Cargo.toml').Replace("'", "''")
        $target = (Join-Path $env:LOCALAPPDATA 'SwiftBonsai2/build/maintenance').Replace("'", "''")
        $message += "`nTo prepare the default tool, run this command from the repository root:`n" +
            "cargo --config 'target.aarch64-pc-windows-msvc.rustflags=[`"-C`",`"target-feature=+crt-static`"]' build --locked --release --target aarch64-pc-windows-msvc --manifest-path '$manifest' --package bonsai-tools --bin bonsai-tools --target-dir '$target'"
    }
    $message += "`nSee docs/development-guide.md#native-chat-app for setup instructions."
    throw $message
}
$resolvedTool = Resolve-Path -LiteralPath $ToolPath
if ($resolvedTool.Provider.Name -cne 'FileSystem' -or [IO.Path]::GetExtension($resolvedTool.ProviderPath) -ine '.exe') {
    throw 'The native launcher must be a filesystem executable (.exe).'
}
$ToolPath = $resolvedTool.ProviderPath

$launchArguments = @('chat', '--repository-root', (Split-Path $PSScriptRoot -Parent))
if ($DataDirectory) { $launchArguments += @('--data-directory', $DataDirectory) }
if ($NativeEngine) { $launchArguments += '--native-engine' }
if ($PublisherEngine) { $launchArguments += '--publisher-engine' }
if ($RequireBrokerPipe) { $launchArguments += '--require-broker-pipe' }
if ($PSBoundParameters.ContainsKey('NativeRowWorkers')) {
    $launchArguments += @('--native-row-workers', [string]$NativeRowWorkers)
}
if ($UseRepackedPtq1) { $launchArguments += '--use-repacked-ptq1' }
if ($UseOriginalPq2) { $launchArguments += '--use-original-pq2' }
if ($GraphicsBackend) { $launchArguments += @('--graphics-backend', $GraphicsBackend.ToLowerInvariant()) }
if ($PromptReuse) { $launchArguments += @('--prompt-reuse', $PromptReuse.ToLowerInvariant()) }
if ($PSBoundParameters.ContainsKey('PromptReuseCheckpoints')) {
    $launchArguments += @('--prompt-reuse-checkpoints', [string]$PromptReuseCheckpoints)
}
if ($PSBoundParameters.ContainsKey('PromptReuseCheckpointBytes')) {
    $launchArguments += @('--prompt-reuse-checkpoint-bytes', [string]$PromptReuseCheckpointBytes)
}
if ($KvCache) { $launchArguments += @('--kv-cache', $KvCache.ToLowerInvariant()) }
if ($PSBoundParameters.ContainsKey('CpuPrefillTokens')) {
    $launchArguments += @('--cpu-prefill-tokens', [string]$CpuPrefillTokens)
}
if ($CpuPrefillGdn) { $launchArguments += @('--cpu-prefill-gdn', $CpuPrefillGdn.ToLowerInvariant()) }
if ($SpecType) { $launchArguments += @('--spec-type', $SpecType.ToLowerInvariant()) }
if ($PSBoundParameters.ContainsKey('DraftMax')) {
    $launchArguments += @('--draft-max', [string]$DraftMax)
}
if ($CpuFamily) { $launchArguments += @('--cpu-family', $CpuFamily.ToLowerInvariant()) }
if ($Placement) { $launchArguments += @('--placement', $Placement.ToLowerInvariant()) }
& $ToolPath @launchArguments
exit $LASTEXITCODE
