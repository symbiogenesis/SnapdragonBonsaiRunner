#Requires -Version 7.6
<# Admit a prepared checker; recover its cache once under a verified native supervisor. #>
[CmdletBinding()]
param(
    [string]$Root = (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent),
    [string]$TargetDirectory = (Join-Path ([IO.Path]::GetTempPath()) 'Bonsai-Push-Sanity-Cargo'),
    [switch]$Prepare
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
function Test-PushHelperReceipt([string]$ReceiptPath, [string]$Image, [string]$SourceIdentity) {
    if (-not (Test-Path -LiteralPath $Image -PathType Leaf) -or -not (Test-Path -LiteralPath $ReceiptPath -PathType Leaf)) { return $false }
    try {
        $receipt = Get-Content -LiteralPath $ReceiptPath -Raw | ConvertFrom-Json -AsHashtable
        return $receipt -is [Collections.IDictionary] -and $receipt.Count -eq 3 -and
            $receipt.Contains('SchemaVersion') -and $receipt.Contains('SourceIdentity') -and $receipt.Contains('Sha256') -and
            $receipt.SchemaVersion -is [long] -and $receipt.SchemaVersion -eq 1 -and
            $receipt.SourceIdentity -is [string] -and $receipt.Sha256 -is [string] -and
            $receipt.SourceIdentity -ceq $SourceIdentity -and
            $receipt.Sha256 -ceq (Get-FileHash -LiteralPath $Image -Algorithm SHA256).Hash.ToLowerInvariant()
    } catch { return $false }
}
$updates = if ([Console]::IsInputRedirected) { [Console]::In.ReadToEnd() } else { '' }
if (-not $updates.Trim() -and -not $Prepare) { return }
$started = [Diagnostics.Stopwatch]::StartNew()
$originalPwsh = $env:BONSAI_PUSH_PWSH_PATH
$originalChecksum = $env:CARGO_UNSTABLE_CHECKSUM_FRESHNESS
$originalFingerprint = $env:CARGO_BUILD_FINGERPRINT
$originalIncremental = $env:CARGO_INCREMENTAL
$env:BONSAI_PUSH_PWSH_PATH = [Environment]::ProcessPath
Add-Type -AssemblyName System.Windows.Forms
$power = [Windows.Forms.SystemInformation]::PowerStatus
Write-Host "Push sanity power before: AC=$($power.PowerLineStatus); charge=$($power.BatteryLifePercent); flags=$($power.BatteryChargeStatus); battery saver unknown."
$target = [IO.Path]::GetFullPath($TargetDirectory)
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('bonsai-push-bootstrap-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($scratch)
try {
    $head = (git -C $Root rev-parse --verify HEAD).Trim()
    $inputs = @('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', '.cargo', 'tools/ci', 'src/artifacts', 'src/windows-process', 'src/windows-credentials', 'scripts/release/Release-Common.ps1')
    $revisions = @($inputs | ForEach-Object { '{0}:{1}' -f $head, $_ })
    $objects = @(git -C $Root rev-parse @revisions)
    if ($objects.Count -ne $inputs.Count -or @($objects | Where-Object { $_ -cnotmatch '^(?:[0-9a-f]{40}|[0-9a-f]{64})$' }).Count) { throw 'Push blocked: committed checker input identities are incomplete.' }
    $sourceId = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes(($objects -join "`n")))).ToLowerInvariant()
    $tool = Join-Path $target 'aarch64-pc-windows-msvc/debug/bonsai-ci.exe'
    $receiptPath = Join-Path $target 'push-checker.json'
    $recoveryTool = Join-Path $target 'aarch64-pc-windows-msvc/debug/bonsai-push-recovery.exe'
    $recoveryReceiptPath = Join-Path $target 'push-recovery.json'
    $recover = -not $Prepare -and -not (Test-PushHelperReceipt $receiptPath $tool $sourceId)
    if ($Prepare -or $recover) {
        $recoveryInputs = @('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', '.cargo', 'src/windows-process')
        $recoveryObjects = @($recoveryInputs | ForEach-Object { $objects[[array]::IndexOf($inputs, $_)] })
        $recoveryId = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes(($recoveryObjects -join "`n")))).ToLowerInvariant()
    }
    # A failed explicit rebuild cannot leave an earlier preparation admitted.
    # Keep its binary for diagnosis/reuse, but publish a receipt only on success.
    if ($Prepare) {
        [void][IO.Directory]::CreateDirectory($target)
        [IO.File]::Delete($receiptPath)
        [IO.File]::Delete($recoveryReceiptPath)
    }
    if ($recover) {
        [IO.File]::Delete($receiptPath)
        if (-not (Test-PushHelperReceipt $recoveryReceiptPath $recoveryTool $recoveryId)) { throw 'Push blocked: recovery supervisor is missing, stale or invalid; prepare with tools/ci/check_push_sanity.ps1 -Prepare in a declared source-build allowance.' }
        Write-Host 'Push checker cache is missing, stale or invalid; attempting one bounded offline incremental recovery.'
    }
    # A valid checker needs only committed checking policy. Both preparation
    # and recovery snapshot the complete committed Cargo input closure.
    $archive = Join-Path $scratch 'policy.zip'
    $policy = Join-Path $scratch 'source'
    $archiveArguments = @('-C', $Root, 'archive', '--format=zip', "--output=$archive", $head)
    if (-not $Prepare -and -not $recover) {
        $archiveArguments += @('--',
            'tools/ci/check_push_snapshot.ps1',
            'tools/ci/check_owned_languages.ps1',
            'tools/ci/check_powershell_syntax.ps1',
            'tools/ci/check_workflow_startup.ps1',
            'tools/ci/check_workflow_syntax.ps1',
            'tools/ci/test_tools_cache_inputs.ps1',
            'tools/ci/test_source_checks.ps1',
            'tools/ci/test_support.ps1')
    }
    git @archiveArguments
    Expand-Archive -LiteralPath $archive -DestinationPath $policy
    if ($Prepare -or $recover) {
        . (Join-Path $policy 'scripts/release/Release-Common.ps1')
        $toolchain = Get-ReleaseToolchain $policy
        if ($Prepare) {
            $env:CARGO_UNSTABLE_CHECKSUM_FRESHNESS = 'true'
            $env:CARGO_BUILD_FINGERPRINT = 'content'
            $env:CARGO_INCREMENTAL = '1'
            Push-Location -LiteralPath $policy
            try {
                # Extra target-only rustc arguments dirty these binary units,
                # even if their cached output bytes were externally changed.
                # Dependencies retain their normal fingerprints and cache.
                $metadata = 'bonsai_push_prepare_' + [guid]::NewGuid().ToString('N')
                cargo "+$toolchain" rustc --quiet --locked --offline --manifest-path (Join-Path $policy 'Cargo.toml') --package bonsai-ci --bin bonsai-ci --target aarch64-pc-windows-msvc --target-dir $target -- -C "metadata=$metadata"
                cargo "+$toolchain" rustc --quiet --locked --offline --manifest-path (Join-Path $policy 'Cargo.toml') --package bonsai-windows-process --bin bonsai-push-recovery --target aarch64-pc-windows-msvc --target-dir $target -- -C "metadata=$metadata"
            }
            finally { Pop-Location }
            @{ SchemaVersion = 1; SourceIdentity = $recoveryId; Sha256 = (Get-FileHash -LiteralPath $recoveryTool -Algorithm SHA256).Hash.ToLowerInvariant() } | ConvertTo-Json | Set-Content -LiteralPath $recoveryReceiptPath -Encoding utf8NoBOM
        } else {
            & $recoveryTool --source $policy --target-directory $target --toolchain $toolchain --bootstrap-milliseconds $started.ElapsedMilliseconds
            if ($LASTEXITCODE) { throw 'Push blocked: bonsai-push-recovery failed; checker receipt remains invalid.' }
        }
        @{ SchemaVersion = 1; SourceIdentity = $sourceId; Sha256 = (Get-FileHash -LiteralPath $tool -Algorithm SHA256).Hash.ToLowerInvariant() } | ConvertTo-Json | Set-Content -LiteralPath $receiptPath -Encoding utf8NoBOM
        if (-not (Test-PushHelperReceipt $receiptPath $tool $sourceId)) {
            [IO.File]::Delete($receiptPath)
            throw 'Push blocked: rebuilt checker did not pass receipt verification.'
        }
    }
    if ($Prepare) {
        'Prepared the committed native checker and recovery supervisor; this is not an outgoing-tip validation verdict.'
        return
    }
    $updates | & $tool push-sanity --root $Root --policy-root $policy --bootstrap-milliseconds $started.ElapsedMilliseconds
    if ($LASTEXITCODE) { throw 'Push blocked: source sanity did not pass.' }
} finally {
    $temporary = [IO.Path]::TrimEndingDirectorySeparator([IO.Path]::GetFullPath([IO.Path]::GetTempPath())) + [IO.Path]::DirectorySeparatorChar
    $absolute = [IO.Path]::GetFullPath($scratch)
    if (-not $absolute.StartsWith($temporary, [StringComparison]::OrdinalIgnoreCase) -or -not [IO.Path]::GetFileName($absolute).StartsWith('bonsai-push-bootstrap-', [StringComparison]::Ordinal)) { throw 'Push bootstrap cleanup escaped temporary storage.' }
    Remove-Item -LiteralPath $absolute -Recurse -Force
    $power = [Windows.Forms.SystemInformation]::PowerStatus
    Write-Host "Push sanity power after: AC=$($power.PowerLineStatus); charge=$($power.BatteryLifePercent); flags=$($power.BatteryChargeStatus); battery saver unknown; aggregate elapsed=$($started.Elapsed)."
    if ($null -eq $originalPwsh) { [Environment]::SetEnvironmentVariable('BONSAI_PUSH_PWSH_PATH', [NullString]::Value) }
    else { $env:BONSAI_PUSH_PWSH_PATH = $originalPwsh }
    if ($null -eq $originalChecksum) { [Environment]::SetEnvironmentVariable('CARGO_UNSTABLE_CHECKSUM_FRESHNESS', [NullString]::Value) }
    else { $env:CARGO_UNSTABLE_CHECKSUM_FRESHNESS = $originalChecksum }
    if ($null -eq $originalFingerprint) { [Environment]::SetEnvironmentVariable('CARGO_BUILD_FINGERPRINT', [NullString]::Value) }
    else { $env:CARGO_BUILD_FINGERPRINT = $originalFingerprint }
    if ($null -eq $originalIncremental) { [Environment]::SetEnvironmentVariable('CARGO_INCREMENTAL', [NullString]::Value) }
    else { $env:CARGO_INCREMENTAL = $originalIncremental }
}
