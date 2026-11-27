#Requires -Version 7.6
# Exercise native stdout/stderr handling without Cargo, compilation or networking.
[CmdletBinding()]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Release-Common.ps1')
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('SwiftBonsai-Adapter-Controls-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($fixture)
$probePath = Join-Path $fixture $(if ($IsWindows) { 'probe.cmd' } else { 'probe.ps1' })
$savedTools = $env:BONSAI_TOOLS_PATH
$savedRustup = $env:RUSTUP_TOOLCHAIN
$savedLocation = Get-Location
$savedPreference = $PSNativeCommandUseErrorActionPreference
$savedExitCode = Get-Variable -Name LASTEXITCODE -Scope Global -ValueOnly -ErrorAction SilentlyContinue
$savedError = [Console]::Error
$capturedError = [IO.StringWriter]::new()
$engine = (Get-Process -Id $PID).Path
$probeMode = 'success'
$repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$expectedCompiler = '+nightly-2026-11-26-aarch64-pc-windows-msvc'
function Invoke-AdapterProbe {
    # Keep the outer adapter's actual native stream redirection. Only executable
    # selection is replaced, and the child never executes the supplied Cargo args.
    if ($IsWindows) {
        & cmd.exe /d /c $probePath $probeMode
    } else {
        & $engine -NoLogo -NoProfile -NonInteractive -File $probePath $probeMode
    }
    $global:LASTEXITCODE = $LASTEXITCODE
}
function cargo {
    if ($args.Count -lt 2 -or $args[0] -cne $expectedCompiler -or $args[1] -cne 'run') {
        throw 'Release Cargo fallback did not explicitly select the repository compiler.'
    }
    if ($probeMode -ceq 'arguments-only') { $global:LASTEXITCODE = 0; return 'null' }
    Invoke-AdapterProbe
}
try {
    if ($IsWindows) {
        @'
@echo off
>&2 echo info: fixture native stderr
if "%~1"=="failure" (
    echo fixture failure stdout
    exit /b 7
)
if "%~1"=="invalid" (
    echo invalid fixture stdout
    exit /b 0
)
echo {"value":"fixture reply","items":[1,2]}
'@ | Set-Content -LiteralPath $probePath -Encoding ascii
    } else {
        @'
param([string]$Mode)
# This stream-only fixture must not wait for inherited stdin to reach EOF.
[Console]::Error.WriteLine('info: fixture native stderr')
if ($Mode -ceq 'failure') {
    [Console]::Out.WriteLine('fixture failure stdout')
    exit 7
}
if ($Mode -ceq 'invalid') {
    [Console]::Out.WriteLine('invalid fixture stdout')
    exit 0
}
[Console]::Out.WriteLine('{"value":"fixture reply","items":[1,2]}')
'@ | Set-Content -LiteralPath $probePath -Encoding utf8NoBOM
    }
    # Match hosted callers that execute in runner.temp with a different default.
    Set-Location -LiteralPath $fixture
    $env:RUSTUP_TOOLCHAIN = 'stable'
    [Console]::SetError($capturedError)
    $PSNativeCommandUseErrorActionPreference = $true
    foreach ($selection in @('configured', 'cargo')) {
        $env:BONSAI_TOOLS_PATH = if ($selection -ceq 'configured') { 'Invoke-AdapterProbe' } else { $null }
        $probeMode = 'success'
        $beforeError = $capturedError.GetStringBuilder().Length
        $result = Invoke-NativeReleasePolicy @{operation='fixture'}
        if ($result.value -cne 'fixture reply' -or @($result.items).Count -ne 2) { throw "Native JSON changed for $selection." }
        if ($capturedError.ToString().Substring($beforeError) -notlike '*info: fixture native stderr*') { throw "Successful stderr was lost for $selection." }
        if (-not $PSNativeCommandUseErrorActionPreference) { throw 'Native error preference was not restored after success.' }
        $probeMode = 'failure'
        $rejected = $false
        try { $null = Invoke-NativeReleasePolicy @{operation='fixture'} }
        catch {
            if ($_.Exception.Message -notlike '*fixture failure stdout*' -or $_.Exception.Message -notlike '*info: fixture native stderr*') { throw }
            $rejected = $true
        }
        if (-not $rejected) { throw "Nonzero exit was accepted for $selection." }
        if (-not $PSNativeCommandUseErrorActionPreference) { throw 'Native error preference was not restored after failure.' }
        $probeMode = 'invalid'
        $rejected = $false
        try { $null = Invoke-NativeReleasePolicy @{operation='fixture'} }
        catch { $rejected = $true }
        if (-not $rejected) { throw "Invalid JSON was accepted for $selection." }
    }
    $probeMode = 'arguments-only'
    $null = & (Join-Path $repository '.github/release/Collect-Notices.ps1') -RepositoryRoot $repository -OutputPath (Join-Path $fixture 'notices.txt')
    $null = & (Join-Path $repository '.github/release/Package-Release.ps1') -RepositoryRoot $repository -InputDirectory $fixture -OutputDirectory $fixture -EvidenceDirectory $fixture -Tag 'v0.3.1' -SourceRevision ('a' * 40)
    $null = & (Join-Path $PSScriptRoot 'Test-WindowsExecutable.ps1') -ExecutablePath (Join-Path $fixture 'inert.exe') -ManifestSha256 ('b' * 64) -EvidenceDirectory $fixture
    Write-Output 'Release adapter controls passed: native JSON and visible stderr, failure diagnostics, and repository-pinned Cargo fallbacks outside the checkout.'
} finally {
    [Console]::SetError($savedError)
    $capturedError.Dispose()
    $env:BONSAI_TOOLS_PATH = $savedTools
    $env:RUSTUP_TOOLCHAIN = $savedRustup
    Set-Location -LiteralPath $savedLocation.Path
    $PSNativeCommandUseErrorActionPreference = $savedPreference
    if ($null -eq $savedExitCode) {
        Remove-Variable -Name LASTEXITCODE -Scope Global -ErrorAction SilentlyContinue
    } else {
        $global:LASTEXITCODE = $savedExitCode
    }
    Remove-Item -LiteralPath $probePath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $fixture -Force
}
