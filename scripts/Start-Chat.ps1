#Requires -Version 7.6

<#
.SYNOPSIS
Launches native chat with the Rust inference engine development default.
.DESCRIPTION
Automatically builds or refreshes bonsai-tools from this checkout when -ToolPath
and BONSAI_TOOLS_PATH are omitted, then launches native chat. Rust owns startup
and application policy.
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
    [ValidateSet('f16', 'rsq4', 'rsq4kvf', 'rsq4k3v')]
    [string]$KvCache,
    [ValidateSet(8, 16, 32, 64)]
    [int]$CpuPrefillTokens,
    [ValidateSet('sequential', 'wy-f32')]
    [string]$CpuPrefillGdn,
    [ValidateSet('none', 'suffix')]
    [string]$SpecType,
    [ValidateRange(1, 8)]
    [int]$DraftMax,
    [ValidateSet('x1', 'x2')]
    [string]$CpuFamily,
    [ValidateSet('auto', 'cpu', 'vulkan', 'composite')]
    [string]$Placement,
    [ValidateSet('pq2', 'ptq1')]
    [string]$Bonsai25,
    [string]$DaisyRoseLora,
    [switch]$NoDaisyRoseLora,
    [ValidateSet('dx12', 'vulkan', 'gl')]
    [string]$GraphicsBackend,
    [switch]$ForceBuild,
    [string]$DataDirectory,
    [string]$ToolPath = $env:BONSAI_TOOLS_PATH
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$exitCode = 0
$savedPath = [Environment]::GetEnvironmentVariable('PATH')
try {
    $repoRoot = Split-Path -LiteralPath $PSScriptRoot

    $cargoPath = $null
    $cargoCommand = Get-Command cargo.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cargoCommand) {
        $cargoPath = $cargoCommand.Source
    } else {
        $cargoCandidates = @()
        if ($env:CARGO_HOME) {
            $cargoCandidates += Join-Path $env:CARGO_HOME 'bin/cargo.exe'
        }
        if ($env:USERPROFILE) {
            $cargoCandidates += Join-Path $env:USERPROFILE '.cargo/bin/cargo.exe'
        }
        foreach ($candidate in $cargoCandidates) {
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                $cargoPath = $candidate
                $cargoDirectory = Split-Path -LiteralPath $cargoPath
                $env:PATH = if ($env:PATH) {
                    $cargoDirectory + [IO.Path]::PathSeparator + $env:PATH
                } else {
                    $cargoDirectory
                }
                break
            }
        }
    }

    $explicitTool = -not [string]::IsNullOrWhiteSpace($ToolPath)
    if ($explicitTool) {
        $ToolPath = $ToolPath.Trim()
        if (-not (Test-Path -LiteralPath $ToolPath -PathType Leaf)) {
            throw "The selected bonsai-tools executable was not found at '$ToolPath'. Correct -ToolPath or BONSAI_TOOLS_PATH, or remove the override to build and launch automatically."
        }
    } else {
        $maintenanceTarget = if ($env:LOCALAPPDATA) {
            Join-Path $env:LOCALAPPDATA 'SnapdragonBonsai/build/maintenance'
        } elseif ($DataDirectory) {
            Join-Path $DataDirectory 'build/maintenance'
        } else {
            $null
        }

        if ($cargoPath -and $maintenanceTarget) {
            $defaultTool = Join-Path $maintenanceTarget 'aarch64-pc-windows-msvc/release/bonsai-tools.exe'
            $hadCachedTool = Test-Path -LiteralPath $defaultTool -PathType Leaf
            if (-not $hadCachedTool) {
                Write-Host 'Building bonsai-tools (first-time setup; subsequent runs reuse this build)...'
            }

            $savedChecksum = [Environment]::GetEnvironmentVariable('CARGO_UNSTABLE_CHECKSUM_FRESHNESS')
            $savedFingerprint = [Environment]::GetEnvironmentVariable('CARGO_BUILD_FINGERPRINT')
            $savedToolchain = [Environment]::GetEnvironmentVariable('RUSTUP_TOOLCHAIN')
            $pushedLocation = $false
            $buildExitCode = 0
            try {
                Push-Location -LiteralPath $repoRoot
                $pushedLocation = $true
                [Environment]::SetEnvironmentVariable('CARGO_UNSTABLE_CHECKSUM_FRESHNESS', 'true')
                [Environment]::SetEnvironmentVariable('CARGO_BUILD_FINGERPRINT', 'content')
                [Environment]::SetEnvironmentVariable('RUSTUP_TOOLCHAIN', [NullString]::Value)

                & $cargoPath --config 'target.aarch64-pc-windows-msvc.rustflags=["-C","target-feature=+crt-static"]' build --locked --release --target aarch64-pc-windows-msvc --manifest-path (Join-Path $repoRoot 'Cargo.toml') --package bonsai-tools --bin bonsai-tools --target-dir $maintenanceTarget
                $buildExitCode = if ($null -ne $LASTEXITCODE) { $LASTEXITCODE } else { 1 }
            } finally {
                if ($pushedLocation) { Pop-Location }
                [Environment]::SetEnvironmentVariable('CARGO_UNSTABLE_CHECKSUM_FRESHNESS', $(if ($null -eq $savedChecksum) { [NullString]::Value } else { $savedChecksum }))
                [Environment]::SetEnvironmentVariable('CARGO_BUILD_FINGERPRINT', $(if ($null -eq $savedFingerprint) { [NullString]::Value } else { $savedFingerprint }))
                [Environment]::SetEnvironmentVariable('RUSTUP_TOOLCHAIN', $(if ($null -eq $savedToolchain) { [NullString]::Value } else { $savedToolchain }))
            }

            if ($buildExitCode -ne 0) {
                if ($hadCachedTool) {
                    throw "Failed to build bonsai-tools (exit code $buildExitCode). If Swift Bonsai 2 is currently running, close it before rebuilding, or pass -ToolPath to use an existing executable."
                }
                throw "Failed to build bonsai-tools (exit code $buildExitCode). See docs/development-guide.md#native-chat-app for setup instructions."
            }

            $ToolPath = $defaultTool
            if (-not $hadCachedTool) {
                Write-Host 'Starting Swift Bonsai 2 (first launch also builds the native UI and broker before opening the window)...'
            }
        } else {
            if ($env:LOCALAPPDATA) {
                foreach ($relative in @(
                    'SnapdragonBonsai/build/maintenance/aarch64-pc-windows-msvc/release/bonsai-tools.exe',
                    'SnapdragonBonsai/build/maintenance/release/bonsai-tools.exe',
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
                throw 'Could not find a prebuilt bonsai-tools.exe or a Rust toolchain (cargo.exe) to build it. Install Rustup (https://rustup.rs) or supply a prebuilt executable with -ToolPath or BONSAI_TOOLS_PATH. See docs/development-guide.md#native-chat-app.'
            }
        }
    }

    if (-not $ToolPath -or -not (Test-Path -LiteralPath $ToolPath -PathType Leaf)) {
        throw 'Could not locate a prebuilt or newly built bonsai-tools executable.'
    }
    $resolvedTool = Resolve-Path -LiteralPath $ToolPath
    if ($resolvedTool.Provider.Name -cne 'FileSystem' -or [IO.Path]::GetExtension($resolvedTool.ProviderPath) -ine '.exe') {
        throw 'The native launcher must be a filesystem executable (.exe).'
    }
    $ToolPath = $resolvedTool.ProviderPath

    $launchArguments = @('chat', '--repository-root', $repoRoot)
    if ($DataDirectory) { $launchArguments += @('--data-directory', $DataDirectory) }
    if ($ForceBuild) { $launchArguments += '--force-build' }
    if ($NativeEngine) { $launchArguments += '--native-engine' }
    if ($PublisherEngine) { $launchArguments += '--publisher-engine' }
    if ($RequireBrokerPipe) { $launchArguments += '--require-broker-pipe' }
    if ($PSBoundParameters.ContainsKey('NativeRowWorkers')) {
        $launchArguments += @('--native-row-workers', [string]$NativeRowWorkers)
    }
    if ($Bonsai25) { $launchArguments += @('--bonsai-2-5', $Bonsai25.ToLowerInvariant()) }
    if ($DaisyRoseLora) { $launchArguments += @('--daisyrose-lora', $DaisyRoseLora) }
    if ($NoDaisyRoseLora) { $launchArguments += '--no-daisyrose-lora' }
    if ($GraphicsBackend) { $launchArguments += @('--graphics-backend', $GraphicsBackend.ToLowerInvariant()) }
    $nativeActive = -not $PublisherEngine
    $suffixSpec = $SpecType -and ($SpecType.ToLowerInvariant() -eq 'suffix')
    $effectivePromptReuse = if ($PSBoundParameters.ContainsKey('PromptReuse')) {
        $PromptReuse.ToLowerInvariant()
    } elseif ($nativeActive -and -not $suffixSpec) {
        'prompt-end-and-latest'
    } else {
        $null
    }
    if ($effectivePromptReuse) { $launchArguments += @('--prompt-reuse', $effectivePromptReuse) }
    if ($PSBoundParameters.ContainsKey('PromptReuseCheckpoints')) {
        $launchArguments += @('--prompt-reuse-checkpoints', [string]$PromptReuseCheckpoints)
    }
    if ($PSBoundParameters.ContainsKey('PromptReuseCheckpointBytes')) {
        $launchArguments += @('--prompt-reuse-checkpoint-bytes', [string]$PromptReuseCheckpointBytes)
    }
    if ($KvCache) { $launchArguments += @('--kv-cache', $KvCache.ToLowerInvariant()) }
    if ($PSBoundParameters.ContainsKey('CpuPrefillTokens')) {
        $launchArguments += @('--cpu-prefill-tokens', [string]$CpuPrefillTokens)
    } elseif ($nativeActive -and -not $suffixSpec) {
        $launchArguments += @('--cpu-prefill-tokens', '16')
    }
    if ($CpuPrefillGdn) {
        $launchArguments += @('--cpu-prefill-gdn', $CpuPrefillGdn.ToLowerInvariant())
    } elseif ($nativeActive -and -not $suffixSpec) {
        $launchArguments += @('--cpu-prefill-gdn', 'wy-f32')
    }
    if ($SpecType) { $launchArguments += @('--spec-type', $SpecType.ToLowerInvariant()) }
    if ($PSBoundParameters.ContainsKey('DraftMax')) {
        $launchArguments += @('--draft-max', [string]$DraftMax)
    }
    if ($CpuFamily) { $launchArguments += @('--cpu-family', $CpuFamily.ToLowerInvariant()) }
    if ($Placement) { $launchArguments += @('--placement', $Placement.ToLowerInvariant()) }
    & $ToolPath @launchArguments
    $exitCode = if ($null -ne $LASTEXITCODE) { $LASTEXITCODE } else { 0 }
} finally {
    [Environment]::SetEnvironmentVariable('PATH', $(if ($null -eq $savedPath) { [NullString]::Value } else { $savedPath }))
}
exit $exitCode
