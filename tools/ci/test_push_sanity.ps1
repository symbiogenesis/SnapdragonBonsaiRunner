#Requires -Version 7.6
param([string[]]$Case, [string]$ReportPath)
$Case = @($Case | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
if (-not [Console]::IsInputRedirected) {
    $forward = @('-NoLogo', '-NoProfile', '-NonInteractive', '-File', $PSCommandPath)
    if ($Case -and $Case.Count -gt 0) { $forward += @('-Case', ($Case -join ',')) }
    if ($ReportPath) { $forward += @('-ReportPath', $ReportPath) }
    '' | & ([Environment]::ProcessPath) @forward
    exit $LASTEXITCODE
}
. (Join-Path $PSScriptRoot 'test_support.ps1')
Add-Type -AssemblyName System.IO.Compression.FileSystem
$adapter = Join-Path $PSScriptRoot 'check_push_sanity.ps1'
$script:PushTemplateCache = @{}

function Get-CiFileSha256([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    try { [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($stream)).ToLowerInvariant() }
    finally { $stream.Dispose() }
}

function Copy-CiTree([string]$Source, [string]$Destination) {
    $stack = [Collections.Generic.Stack[string[]]]::new()
    $stack.Push(@($Source, $Destination))
    while ($stack.Count -gt 0) {
        $pair = $stack.Pop()
        $src = $pair[0]; $dst = $pair[1]
        [void][IO.Directory]::CreateDirectory($dst)
        foreach ($file in [IO.Directory]::GetFiles($src)) {
            [IO.File]::Copy($file, (Join-Path $dst ([IO.Path]::GetFileName($file))), $true)
        }
        foreach ($dir in [IO.Directory]::GetDirectories($src)) {
            $stack.Push(@($dir, (Join-Path $dst ([IO.Path]::GetFileName($dir)))))
        }
    }
}

function Get-PushRecoveryTemplate([bool]$WithPreparation) {
    $key = if ($WithPreparation) { 'prep' } else { 'plain' }
    if ($script:PushTemplateCache.ContainsKey($key)) { return $script:PushTemplateCache[$key] }
    $scratch = New-CiScratch 'bonsai-push-template-'
    $repository = Join-Path $scratch 'repository'
    $checkerInputs = @('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', '.cargo', 'tools/ci', 'src/artifacts', 'src/windows-process', 'src/windows-credentials', 'scripts/release/Release-Common.ps1')
    $recoveryInputs = @('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', '.cargo', 'src/windows-process')
    foreach ($path in @('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', '.cargo/fixture.txt', 'tools/ci/fixture.txt', 'src/artifacts/fixture.txt', 'src/windows-process/fixture.txt', 'src/windows-credentials/fixture.txt')) { Write-CiFixture $repository $path 'Inert committed build identity; no Cargo consumer.' }
    Write-CiFixture $repository 'unrelated-source.txt' 'Committed full-source input.'
    Write-CiFixture $repository 'scripts/release/Release-Common.ps1' "function Get-ReleaseToolchain([string]`$RepositoryRoot) { 'nightly-2026-11-26-aarch64-pc-windows-msvc' }"
    if ($WithPreparation) {
        Write-CiFixture $repository 'tools/ci/prepare_push_helpers.ps1' @'
#Requires -Version 7.6
param([string]$Root,[string]$SourceRoot,[string]$TargetDirectory,[string]$ScratchDirectory,[long]$BootstrapMilliseconds)
$global:BonsaiPushAutomaticPreparationCalls++
Assert-CiEqual $global:BonsaiPushAutomaticPreparationCalls 1
Assert-CiEqual $Root $global:BonsaiPushPreparationExpectedRoot
Assert-CiEqual $TargetDirectory $global:BonsaiPushPreparationExpectedTarget
Assert-CiEqual $SourceRoot ([IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..')))
Assert-CiEqual $ScratchDirectory (Split-Path $SourceRoot -Parent)
Assert-Ci ($BootstrapMilliseconds -gt 0) 'Automatic preparation did not receive elapsed bootstrap time.'
Assert-Ci (-not (Test-Path -LiteralPath (Join-Path $TargetDirectory 'push-checker.json'))) 'Preparation retained the old checker receipt.'
Assert-Ci (-not (Test-Path -LiteralPath (Join-Path $TargetDirectory 'push-recovery.json'))) 'Preparation retained the old recovery receipt.'
Assert-CiEqual ([IO.File]::ReadAllText((Join-Path $SourceRoot 'unrelated-source.txt'))) 'Committed full-source input.'
if ($global:BonsaiPushPreparationState -ceq 'handoff') { throw 'Expected committed automatic preparation invocation before native execution.' }
foreach ($entry in @(@{Receipt='push-checker.json';Image='bonsai-ci.exe'},@{Receipt='push-recovery.json';Image='bonsai-push-recovery.exe'})) {
    $image=Join-Path $TargetDirectory "aarch64-pc-windows-msvc/debug/$($entry.Image)"
    @{SchemaVersion=1;SourceIdentity=('0' * 64);Sha256=(Get-FileHash -LiteralPath $image -Algorithm SHA256).Hash.ToLowerInvariant()} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $TargetDirectory $entry.Receipt) -Encoding utf8NoBOM
}
'@
    }
    & git -C $repository init --quiet
    if ($LASTEXITCODE) { throw 'Cannot initialize recovery fixture.' }
    & git -C $repository add --all
    if ($LASTEXITCODE) { throw 'Cannot stage recovery fixture.' }
    & git -C $repository -c user.name=Fixture -c user.email=fixture@example.invalid -c core.hooksPath= -c commit.gpgsign=false commit --quiet -m fixture
    if ($LASTEXITCODE) { throw 'Cannot commit recovery fixture.' }
    $head = (& git -C $repository rev-parse HEAD).Trim()
    $identities = @{}
    foreach ($selection in @(@{Name = 'checker'; Inputs = $checkerInputs }, @{Name = 'recovery'; Inputs = $recoveryInputs })) {
        # Resolve objects via cat-file, independently of the rev-parse adapter.
        $batchInput = ($selection.Inputs | ForEach-Object { '{0}:{1}' -f $head, $_ }) -join "`n"
        $objects = @($batchInput | & git -C $repository cat-file --batch-check='%(objectname)' | ForEach-Object { $_.Trim() })
        if ($LASTEXITCODE -ne 0 -or $objects.Count -ne $selection.Inputs.Count -or @($objects | Where-Object { $_ -cnotmatch '^[0-9a-f]{40,64}$' }).Count -gt 0) {
            throw 'Cannot resolve recovery template git objects.'
        }
        $identities[$selection.Name] = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes(($objects -join "`n")))).ToLowerInvariant()
    }
    Write-CiFixture $repository 'unrelated-source.txt' 'Uncommitted poison must not reach recovery.'
    if ($WithPreparation) {
        Write-CiFixture $repository 'tools/ci/prepare_push_helpers.ps1' "throw 'Uncommitted preparation helper must never execute.'"
    }
    $template = @{ Scratch = $scratch; Repo = $repository; Head = $head; Identities = $identities }
    $script:PushTemplateCache[$key] = $template
    return $template
}

function Invoke-RedirectedStdin([string]$InputText, [scriptblock]$Work) {
    $originalIn = [Console]::In
    $reader = [IO.StringReader]::new($InputText)
    try {
        [Console]::SetIn($reader)
        & $Work
    } finally {
        [Console]::SetIn($originalIn)
        $reader.Dispose()
    }
}

function Invoke-PushRecoveryFixture([string]$CheckerState, [string]$RecoveryState = 'valid', [switch]$RunRecovery, [ValidateSet('none', 'handoff', 'invalid-output')][string]$PreparationState = 'none') {
    $scratch = New-CiScratch 'bonsai-push-recovery-'
    try {
        $template = Get-PushRecoveryTemplate ($PreparationState -cne 'none')
        $repository = $template.Repo
        $head = $template.Head
        $identities = $template.Identities
        $target = if ($CheckerState -ceq 'absent-target') { Join-Path $scratch 'absent/cache/target' } else { Join-Path $scratch 'target' }
        $checker = Join-Path $target 'aarch64-pc-windows-msvc/debug/bonsai-ci.exe'
        $recovery = Join-Path $target 'aarch64-pc-windows-msvc/debug/bonsai-push-recovery.exe'
        if ($CheckerState -cne 'absent-target') {
            Write-CiFixture $target 'aarch64-pc-windows-msvc/debug/bonsai-ci.exe' 'Inert checker; it must never execute.'
            Write-CiFixture $target 'aarch64-pc-windows-msvc/debug/bonsai-push-recovery.exe' 'Inert supervisor; archive interception stops before execution.'
            if ($RunRecovery) {
                # The native WHERE utility rejects the supervisor's switches promptly.
                # It cannot compile a checker or manufacture a successful receipt.
                Copy-Item -LiteralPath (Join-Path $env:SystemRoot 'System32/where.exe') -Destination $recovery -Force
            }
            $checkerReceipt = @{SchemaVersion = 1; SourceIdentity = $identities.checker; Sha256 = (Get-CiFileSha256 $checker) }
            if ($CheckerState -ceq 'stale') { $checkerReceipt.SourceIdentity = '0' * 64 }
            if ($CheckerState -ceq 'collection-source') { $checkerReceipt.SourceIdentity = @($identities.checker, ('0' * 64)) }
            if ($CheckerState -ceq 'collection-digest') { $checkerReceipt.Sha256 = @($checkerReceipt.Sha256, ('0' * 64)) }
            $checkerReceipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $target 'push-checker.json') -Encoding utf8NoBOM
            if ($CheckerState -ceq 'missing') { [IO.File]::Delete((Join-Path $target 'push-checker.json')) }
            if ($CheckerState -ceq 'malformed') { Write-CiFixture $target 'push-checker.json' '{"unfinished":' }
            if ($CheckerState -ceq 'corrupt') { [IO.File]::AppendAllText($checker, 'Changed after receipt publication.') }
            $recoveryReceipt = @{SchemaVersion = 1; SourceIdentity = $identities.recovery; Sha256 = (Get-CiFileSha256 $recovery) }
            if ($RecoveryState -ceq 'invalid') { $recoveryReceipt.Sha256 = '0' * 64 }
            if ($RecoveryState -ceq 'stale') { $recoveryReceipt.SourceIdentity = '0' * 64 }
            if ($RecoveryState -ceq 'collection-source') { $recoveryReceipt.SourceIdentity = @($identities.recovery, ('0' * 64)) }
            if ($RecoveryState -ceq 'collection-digest') { $recoveryReceipt.Sha256 = @($recoveryReceipt.Sha256, ('0' * 64)) }
            $recoveryReceipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $target 'push-recovery.json') -Encoding utf8NoBOM
            if ($RecoveryState -ceq 'missing-receipt') { [IO.File]::Delete((Join-Path $target 'push-recovery.json')) }
            if ($RecoveryState -ceq 'missing-image') { [IO.File]::Delete($recovery) }
            if ($RecoveryState -ceq 'corrupt') { [IO.File]::AppendAllText($recovery, 'Changed after recovery receipt publication.') }
        } else {
            Assert-Ci (-not (Test-Path -LiteralPath $target)) 'Absent recovery target already exists.'
        }
        $gitPath = (Get-Command git -CommandType Application | Select-Object -First 1).Source
        try {
            Invoke-RedirectedStdin "refs/heads/main $head refs/heads/main $head`n" {
                $PSNativeCommandUseErrorActionPreference = $true
                $global:BonsaiPushRecoveryArchiveCalls = 0
                $global:BonsaiPushAutomaticPreparationCalls = 0
                $global:BonsaiPushPreparationState = $PreparationState
                $global:BonsaiPushPreparationExpectedRoot = $repository
                $global:BonsaiPushPreparationExpectedTarget = $target
                function git {
                    & $gitPath @args
                    if ($LASTEXITCODE) { throw 'Fixture Git command failed.' }
                    if ($args -ccontains 'archive') {
                        $global:BonsaiPushRecoveryArchiveCalls++
                        Assert-CiEqual $global:BonsaiPushRecoveryArchiveCalls 1
                        Assert-Ci (-not (Test-Path -LiteralPath (Join-Path $target 'push-checker.json'))) 'Recovery retained the old checker receipt during source intake.'
                        if ($RecoveryState -cne 'valid') {
                            Assert-Ci (-not (Test-Path -LiteralPath (Join-Path $target 'push-recovery.json'))) 'Automatic preparation retained the invalid recovery receipt during source intake.'
                        }
                        Assert-Ci (-not ($args -ccontains '--')) 'Recovery selected a partial policy archive.'
                        $archive = @($args | Where-Object { $_.StartsWith('--output=', [StringComparison]::Ordinal) })[0].Substring(9)
                        $unpacked = Join-Path $scratch 'unpacked'
                        [IO.Compression.ZipFile]::ExtractToDirectory($archive, $unpacked, $true)
                        Assert-CiEqual ([IO.File]::ReadAllText((Join-Path $unpacked 'unrelated-source.txt'))) 'Committed full-source input.'
                        Assert-Ci (Test-Path -LiteralPath (Join-Path $unpacked 'Cargo.toml')) 'Recovery omitted committed build inputs.'
                        if ($RecoveryState -cne 'valid') {
                            if ($PreparationState -ceq 'none') { throw 'Expected automatic preparation from complete committed source before native execution.' }
                        } elseif (-not $RunRecovery) { throw 'Expected complete committed recovery source verified before supervisor execution.' }
                    }
                }
                $failure = $null
                try { & $adapter -Root $repository -TargetDirectory $target }
                catch { $failure = $_.Exception.Message }
                Assert-Ci ($null -ne $failure) 'Recovery fixture unexpectedly admitted an inert checker.'
                Assert-Ci (-not (Test-Path -LiteralPath (Join-Path $target 'push-checker.json'))) 'Failed recovery published or retained a checker receipt.'
                if ($RecoveryState -cne 'valid') {
                    Assert-CiEqual $global:BonsaiPushRecoveryArchiveCalls 1
                    if ($PreparationState -ceq 'handoff') {
                        Assert-CiEqual $global:BonsaiPushAutomaticPreparationCalls 1
                        Assert-CiEqual $failure 'Expected committed automatic preparation invocation before native execution.'
                    } elseif ($PreparationState -ceq 'invalid-output') {
                        Assert-CiEqual $global:BonsaiPushAutomaticPreparationCalls 1
                        Assert-Ci ($failure.Contains('automatic preparation did not produce verified helpers for the original committed source')) 'Automatic preparation admitted helpers for a different source identity.'
                    } else {
                        Assert-CiEqual $global:BonsaiPushAutomaticPreparationCalls 0
                        Assert-CiEqual $failure 'Expected automatic preparation from complete committed source before native execution.'
                    }
                    Assert-Ci (-not (Test-Path -LiteralPath (Join-Path $target 'push-recovery.json'))) 'Failed automatic preparation published or retained a recovery receipt.'
                } elseif ($RunRecovery) {
                    Assert-CiEqual $global:BonsaiPushRecoveryArchiveCalls 1
                    Assert-Ci ($failure.Contains('bonsai-push-recovery')) 'Failure did not originate from the recovery executable.'
                } else {
                    Assert-CiEqual $failure 'Expected complete committed recovery source verified before supervisor execution.'
                }
            }
        } finally {
            Remove-Variable -Name 'BonsaiPush*' -Scope Global -ErrorAction SilentlyContinue
        }
    } finally { Remove-CiScratch $scratch }
}

$cases = [ordered]@{
    stale_checker_recovers_from_committed_source = { Invoke-PushRecoveryFixture 'stale' }
    malformed_checker_receipt_recovers_from_committed_source = { Invoke-PushRecoveryFixture 'malformed' }
    collection_checker_source_recovers_from_committed_source = { Invoke-PushRecoveryFixture 'collection-source' }
    collection_checker_digest_recovers_from_committed_source = { Invoke-PushRecoveryFixture 'collection-digest' }
    corrupt_checker_recovers_from_committed_source = { Invoke-PushRecoveryFixture 'corrupt' }
    missing_checker_receipt_recovers_from_committed_source = { Invoke-PushRecoveryFixture 'missing' }
    unverified_supervisor_triggers_preparation = { Invoke-PushRecoveryFixture 'stale' 'invalid' }
    absent_target_directory_triggers_preparation = { Invoke-PushRecoveryFixture 'absent-target' 'absent-target' -PreparationState handoff }
    missing_recovery_receipt_triggers_preparation = { Invoke-PushRecoveryFixture 'stale' 'missing-receipt' }
    missing_recovery_helper_triggers_preparation = { Invoke-PushRecoveryFixture 'stale' 'missing-image' }
    stale_recovery_source_triggers_preparation = { Invoke-PushRecoveryFixture 'stale' 'stale' }
    corrupt_recovery_helper_triggers_preparation = { Invoke-PushRecoveryFixture 'stale' 'corrupt' }
    collection_recovery_source_triggers_preparation = { Invoke-PushRecoveryFixture 'stale' 'collection-source' }
    collection_recovery_digest_triggers_preparation = { Invoke-PushRecoveryFixture 'stale' 'collection-digest' }
    automatic_preparation_invokes_committed_helper_once = { Invoke-PushRecoveryFixture 'stale' 'invalid' -PreparationState handoff }
    automatic_preparation_rejects_invalid_prepared_outputs = { Invoke-PushRecoveryFixture 'stale' 'invalid' -PreparationState invalid-output }
    failed_automatic_recovery_invalidates_existing_receipt = { Invoke-PushRecoveryFixture 'stale' -RunRecovery }
    ordinary_push_archives_only_committed_policy = {
        $scratch = New-CiScratch 'bonsai-push-policy-'
        try {
            $repository = Join-Path $scratch 'repository'; $target = Join-Path $scratch 'target'
            $inputs = @('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', '.cargo', 'tools/ci', 'src/artifacts', 'src/windows-process', 'src/windows-credentials', 'scripts/release/Release-Common.ps1')
            foreach ($path in @('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', '.cargo/fixture.txt', 'src/artifacts/fixture.txt', 'src/windows-process/fixture.txt', 'src/windows-credentials/fixture.txt', 'scripts/release/Release-Common.ps1', 'unrelated-source.txt')) { Write-CiFixture $repository $path 'Inert committed source; no compiler consumer.' }
            $policy = @('check_push_snapshot.ps1', 'check_owned_languages.ps1', 'check_powershell_syntax.ps1', 'check_workflow_startup.ps1', 'check_workflow_syntax.ps1', 'native_ci_cache.ps1', 'test_device_concurrency.ps1', 'test_foundation_handoff.ps1', 'test_native_ci_cache.ps1', 'test_rust_toolchain.ps1', 'test_tools_cache_inputs.ps1', 'test_source_checks.ps1', 'test_support.ps1')
            foreach ($name in $policy) { Write-CiFixture $repository "tools/ci/$name" "Committed checking policy: $name" }
            & git -C $repository init --quiet
            if ($LASTEXITCODE) { throw 'Cannot initialize policy archive fixture.' }
            & git -C $repository add --all
            if ($LASTEXITCODE) { throw 'Cannot stage policy archive fixture.' }
            & git -C $repository -c user.name=Fixture -c user.email=fixture@example.invalid -c core.hooksPath= -c commit.gpgsign=false commit --quiet -m fixture
            if ($LASTEXITCODE) { throw 'Cannot commit policy archive fixture.' }
            $head = (& git -C $repository rev-parse HEAD).Trim()
            # Resolve objects via cat-file, independently of the rev-parse adapter.
            $batchInput = ($inputs | ForEach-Object { '{0}:{1}' -f $head, $_ }) -join "`n"
            $objects = @($batchInput | & git -C $repository cat-file --batch-check='%(objectname)' | ForEach-Object { $_.Trim() })
            if ($LASTEXITCODE -ne 0 -or $objects.Count -ne $inputs.Count -or @($objects | Where-Object { $_ -cnotmatch '^[0-9a-f]{40,64}$' }).Count -gt 0) {
                throw 'Cannot resolve policy archive fixture git objects.'
            }
            $sourceId = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes(($objects -join "`n")))).ToLowerInvariant()
            $binary = Join-Path $target 'aarch64-pc-windows-msvc/debug/bonsai-ci.exe'
            Write-CiFixture $target 'aarch64-pc-windows-msvc/debug/bonsai-ci.exe' 'Inert checker; archive interception must stop before execution.'
            @{SchemaVersion = 1; SourceIdentity = $sourceId; Sha256 = (Get-CiFileSha256 $binary) } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $target 'push-checker.json') -Encoding utf8NoBOM
            foreach ($name in $policy) { Write-CiFixture $repository "tools/ci/$name" 'Uncommitted poison must not become checking policy.' }
            $gitPath = (Get-Command git -CommandType Application | Select-Object -First 1).Source
            Invoke-RedirectedStdin "refs/heads/main $head refs/heads/main $head`n" {
                $PSNativeCommandUseErrorActionPreference = $true
                function git {
                    & $gitPath @args
                    if ($LASTEXITCODE) { throw 'Fixture Git command failed.' }
                    if ($args -ccontains 'archive') {
                        $archive = @($args | Where-Object { $_.StartsWith('--output=', [StringComparison]::Ordinal) })[0].Substring(9)
                        $unpacked = Join-Path $scratch 'unpacked'
                        [IO.Compression.ZipFile]::ExtractToDirectory($archive, $unpacked, $true)
                        $expected = @('check_push_snapshot.ps1', 'check_owned_languages.ps1', 'check_powershell_syntax.ps1', 'check_workflow_startup.ps1', 'check_workflow_syntax.ps1', 'native_ci_cache.ps1', 'test_device_concurrency.ps1', 'test_foundation_handoff.ps1', 'test_native_ci_cache.ps1', 'test_rust_toolchain.ps1', 'test_tools_cache_inputs.ps1', 'test_source_checks.ps1', 'test_support.ps1')
                        $actual = @(Get-ChildItem -LiteralPath $unpacked -File -Recurse | ForEach-Object { [IO.Path]::GetRelativePath($unpacked, $_.FullName).Replace('\', '/') })
                        Assert-CiSet $actual @($expected | ForEach-Object { "tools/ci/$_" })
                        foreach ($name in $expected) { Assert-CiEqual ([IO.File]::ReadAllText((Join-Path $unpacked "tools/ci/$name"))) "Committed checking policy: $name" }
                        throw 'Expected committed policy archive verified before helper execution.'
                    }
                }
                try { & $adapter -Root $repository -TargetDirectory $target; throw 'Ordinary push did not reach the archive assertion.' }
                catch { if ($_.Exception.Message -cne 'Expected committed policy archive verified before helper execution.') { throw } }
            }
        } finally { Remove-CiScratch $scratch }
    }
    fresh_preparation_reaches_committed_bootstrap = {
        $scratch = New-CiScratch 'bonsai-push-prepare-fresh-'
        try {
            $repository = Join-Path $scratch 'repository'; $target = Join-Path $scratch 'absent/cache/target'
            foreach ($path in @('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', '.cargo/fixture.txt', 'tools/ci/fixture.txt', 'src/artifacts/fixture.txt', 'src/windows-process/fixture.txt', 'src/windows-credentials/fixture.txt')) { Write-CiFixture $repository $path 'Inert source identity fixture; no compiler consumer.' }
            Write-CiFixture $repository 'scripts/release/Release-Common.ps1' "function Get-ReleaseToolchain([string]`$RepositoryRoot) { throw 'Cold-start reached committed bootstrap; Cargo must not run.' }"
            & git -C $repository init --quiet
            if ($LASTEXITCODE) { throw 'Cannot initialize fresh preparation fixture.' }
            & git -C $repository add --all
            if ($LASTEXITCODE) { throw 'Cannot stage fresh preparation fixture.' }
            & git -C $repository -c user.name=Fixture -c user.email=fixture@example.invalid -c core.hooksPath= -c commit.gpgsign=false commit --quiet -m fixture
            if ($LASTEXITCODE) { throw 'Cannot commit fresh preparation fixture.' }
            Assert-Ci (-not (Test-Path -LiteralPath $target)) 'Fresh preparation target already exists.'
            Assert-Ci ([Console]::IsInputRedirected) 'Preparation fixture requires redirected stdin.'
            $originalInput = [Console]::In
            $unreadableInput = [IO.StringReader]::new('Preparation must ignore this stream.')
            $unreadableInput.Dispose()
            try {
                # Any read fails immediately, exposing open-pipe hangs without hanging CI.
                [Console]::SetIn($unreadableInput)
                Assert-CiReject { & $adapter -Root $repository -TargetDirectory $target -Prepare } 'Cold-start reached committed bootstrap; Cargo must not run.'
            } finally { [Console]::SetIn($originalInput) }
            Assert-Ci (-not (Test-Path -LiteralPath (Join-Path $target 'push-checker.json'))) 'Failed fresh preparation published a receipt.'
        } finally { Remove-CiScratch $scratch }
    }
    failed_preparation_invalidates_existing_receipt = {
        $scratch = New-CiScratch 'bonsai-push-prepare-'
        try {
            $repository = Join-Path $scratch 'repository'; $target = Join-Path $scratch 'target'
            foreach ($path in @('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', '.cargo/fixture.txt', 'tools/ci/fixture.txt', 'src/artifacts/fixture.txt', 'src/windows-process/fixture.txt', 'src/windows-credentials/fixture.txt')) { Write-CiFixture $repository $path 'Inert source identity fixture; no compiler consumer.' }
            Write-CiFixture $repository 'scripts/release/Release-Common.ps1' "function Get-ReleaseToolchain([string]`$RepositoryRoot) { throw 'Expected preparation failure before Cargo.' }"
            & git -C $repository init --quiet
            if ($LASTEXITCODE) { throw 'Cannot initialize push preparation fixture.' }
            & git -C $repository add --all
            if ($LASTEXITCODE) { throw 'Cannot stage push preparation fixture.' }
            & git -C $repository -c user.name=Fixture -c user.email=fixture@example.invalid -c core.hooksPath= -c commit.gpgsign=false commit --quiet -m fixture
            if ($LASTEXITCODE) { throw 'Cannot commit push preparation fixture.' }
            Write-CiFixture $target 'push-checker.json' '{"SchemaVersion":1,"SourceIdentity":"prior-source","Sha256":"prior-bytes"}'
            Write-CiFixture $target 'push-recovery.json' '{"SchemaVersion":1,"SourceIdentity":"prior-recovery-source","Sha256":"prior-recovery-bytes"}'
            $binary = 'Existing helper retained for diagnosis; it must never execute.'
            Write-CiFixture $target 'aarch64-pc-windows-msvc/debug/bonsai-ci.exe' $binary
            Assert-CiReject { & $adapter -Root $repository -TargetDirectory $target -Prepare } 'Expected preparation failure before Cargo.'
            Assert-Ci (-not (Test-Path -LiteralPath (Join-Path $target 'push-checker.json'))) 'A failed rebuild retained its preparation receipt.'
            Assert-Ci (-not (Test-Path -LiteralPath (Join-Path $target 'push-recovery.json'))) 'A failed rebuild retained its recovery receipt.'
            Assert-CiEqual ([IO.File]::ReadAllText((Join-Path $target 'aarch64-pc-windows-msvc/debug/bonsai-ci.exe'))) $binary
        } finally { Remove-CiScratch $scratch }
    }
    cargo_path_fallback_discovers_cargo_and_restores_path = {
        $scratch = New-CiScratch 'bonsai-push-cargo-path-'
        try {
            $repository = Join-Path $scratch 'repository'; $target = Join-Path $scratch 'target'
            foreach ($path in @('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', '.cargo/fixture.txt', 'tools/ci/fixture.txt', 'src/artifacts/fixture.txt', 'src/windows-process/fixture.txt', 'src/windows-credentials/fixture.txt')) { Write-CiFixture $repository $path 'Inert source identity fixture; no compiler consumer.' }
            Write-CiFixture $repository 'scripts/release/Release-Common.ps1' @'
function Get-ReleaseToolchain([string]$RepositoryRoot) {
    $found = Get-Command cargo.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $found) { throw 'Cargo missing from PATH.' }
    throw "Cargo resolved at: $($found.Source)"
}
'@
            & git -C $repository init --quiet
            if ($LASTEXITCODE) { throw 'Cannot initialize cargo-path fixture.' }
            & git -C $repository add --all
            if ($LASTEXITCODE) { throw 'Cannot stage cargo-path fixture.' }
            & git -C $repository -c user.name=Fixture -c user.email=fixture@example.invalid -c core.hooksPath= -c commit.gpgsign=false commit --quiet -m fixture
            if ($LASTEXITCODE) { throw 'Cannot commit cargo-path fixture.' }
            $cargoHome = Join-Path $scratch 'cargo-home'
            $userProfile = Join-Path $scratch 'user-profile'
            $cargoHomeExe = [IO.Path]::GetFullPath((Join-Path $cargoHome 'bin/cargo.exe'))
            $userProfileExe = [IO.Path]::GetFullPath((Join-Path $userProfile '.cargo/bin/cargo.exe'))
            Write-CiFixture $cargoHome 'bin/cargo.exe' 'Inert cargo-home placeholder; must never execute.'
            Write-CiFixture $userProfile '.cargo/bin/cargo.exe' 'Inert user-profile placeholder; must never execute.'
            $gitDirectory = [IO.Path]::GetDirectoryName((Get-Command git -CommandType Application | Select-Object -First 1).Source)
            $cleanPath = $gitDirectory + [IO.Path]::PathSeparator + (Join-Path $env:SystemRoot 'System32')
            $origPath = [Environment]::GetEnvironmentVariable('PATH')
            $origCargoHome = [Environment]::GetEnvironmentVariable('CARGO_HOME')
            $origUserProfile = [Environment]::GetEnvironmentVariable('USERPROFILE')
            try {
                $env:PATH = $cleanPath
                Assert-Ci ($null -eq (Get-Command cargo.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1)) 'Clean fixture PATH unexpectedly contained cargo.exe.'
                [Environment]::SetEnvironmentVariable('CARGO_HOME', $cargoHome)
                [Environment]::SetEnvironmentVariable('USERPROFILE', $userProfile)
                Assert-CiReject { & $adapter -Root $repository -TargetDirectory $target -Prepare } "Cargo resolved at: $cargoHomeExe"
                Assert-Ci ([Environment]::GetEnvironmentVariable('PATH') -ceq $cleanPath) 'Adapter did not restore PATH after CARGO_HOME fallback.'
                [Environment]::SetEnvironmentVariable('CARGO_HOME', [NullString]::Value)
                Assert-CiReject { & $adapter -Root $repository -TargetDirectory $target -Prepare } "Cargo resolved at: $userProfileExe"
                Assert-Ci ([Environment]::GetEnvironmentVariable('PATH') -ceq $cleanPath) 'Adapter did not restore PATH after USERPROFILE fallback.'
            } finally {
                if ($null -eq $origPath) { [Environment]::SetEnvironmentVariable('PATH', [NullString]::Value) } else { [Environment]::SetEnvironmentVariable('PATH', $origPath) }
                if ($null -eq $origCargoHome) { [Environment]::SetEnvironmentVariable('CARGO_HOME', [NullString]::Value) } else { [Environment]::SetEnvironmentVariable('CARGO_HOME', $origCargoHome) }
                if ($null -eq $origUserProfile) { [Environment]::SetEnvironmentVariable('USERPROFILE', [NullString]::Value) } else { [Environment]::SetEnvironmentVariable('USERPROFILE', $origUserProfile) }
            }
        } finally { Remove-CiScratch $scratch }
    }
    push_file_retry_and_atomic_receipt_cleanup = {
        $scratch = New-CiScratch 'bonsai-push-file-retry-'
        try {
            Invoke-RedirectedStdin '' {
                . $adapter
                # 1. Transient IOException & UnauthorizedAccessException recover; ActionPreferenceStopException unwrapped
                $script:attempts = 0
                Invoke-PushFileRetry -MaxAttempts 4 -InitialDelayMs 1 -Operation {
                    $script:attempts++
                    if ($script:attempts -eq 1) { throw [IO.IOException]::new('locked') }
                    if ($script:attempts -eq 2) { throw [UnauthorizedAccessException]::new('denied') }
                }
                Assert-CiEqual $script:attempts 3

                $script:apsAttempts = 0
                Invoke-PushFileRetry -MaxAttempts 3 -InitialDelayMs 1 -Operation {
                    $script:apsAttempts++
                    if ($script:apsAttempts -eq 1) {
                        Write-Error -Exception ([IO.IOException]::new('wrapped')) -Message 'wrapped' -ErrorAction Stop
                    }
                }
                Assert-CiEqual $script:apsAttempts 2

                # Non-transient InvalidOperationException fails on attempt 1 without retry
                $script:nonTransAttempts = 0
                Assert-CiReject {
                    Invoke-PushFileRetry -MaxAttempts 4 -InitialDelayMs 1 -Operation {
                        $script:nonTransAttempts++
                        throw [InvalidOperationException]::new('permanent')
                    }
                } 'permanent'
                Assert-CiEqual $script:nonTransAttempts 1

                # Exhaustion stops at MaxAttempts
                $script:exhAttempts = 0
                Assert-CiReject {
                    Invoke-PushFileRetry -MaxAttempts 3 -InitialDelayMs 1 -Operation {
                        $script:exhAttempts++
                        throw [IO.IOException]::new('always locked')
                    }
                } 'always locked'
                Assert-CiEqual $script:exhAttempts 3

                # 2. Write-PushReceiptAtomic writes UTF-8 without BOM, leaves no .partial, cleans .partial on failure
                $receiptPath = Join-Path $scratch 'sub/push-checker.json'
                Write-PushReceiptAtomic $receiptPath @{ SchemaVersion = 1; Sha256 = ('a' * 64) }
                $bytes = [IO.File]::ReadAllBytes($receiptPath)
                Assert-Ci ($bytes.Length -ge 3 -and -not ($bytes[0] -eq 0xef -and $bytes[1] -eq 0xbb -and $bytes[2] -eq 0xbf)) 'Push receipt must not include UTF-8 BOM.'
                Assert-CiEqual (@(Get-ChildItem -LiteralPath (Split-Path $receiptPath -Parent) -Filter '*.partial').Count) 0

                $dirTarget = Join-Path $scratch 'occupied'
                [void][IO.Directory]::CreateDirectory($dirTarget)
                Assert-CiReject { Write-PushReceiptAtomic $dirTarget @{ SchemaVersion = 1 } }
                Assert-CiEqual (@(Get-ChildItem -LiteralPath $scratch -Filter '*.partial').Count) 0
            }
        } finally { Remove-CiScratch $scratch }
    }
    hooks_path_rejection_restores_pwsh_env = {
        $scratch = New-CiScratch 'bonsai-push-hookspath-env-'
        $origPwsh = [Environment]::GetEnvironmentVariable('BONSAI_PUSH_PWSH_PATH')
        try {
            $repository = Join-Path $scratch 'repository'
            Write-CiFixture $repository '.githooks/pre-push' '# inert hook'
            & git -C $repository init --quiet
            if ($LASTEXITCODE) { throw 'Cannot initialize hooksPath fixture.' }
            & git -C $repository config core.hooksPath wrong-hooks
            if ($LASTEXITCODE) { throw 'Cannot set core.hooksPath fixture.' }
            [Environment]::SetEnvironmentVariable('BONSAI_PUSH_PWSH_PATH', 'sentinel-pwsh-path')
            Invoke-RedirectedStdin "refs/heads/main $('0' * 40) refs/heads/main $('0' * 40)`n" {
                Assert-CiReject { & $adapter -Root $repository } 'core.hooksPath'
            }
            Assert-CiEqual ([Environment]::GetEnvironmentVariable('BONSAI_PUSH_PWSH_PATH')) 'sentinel-pwsh-path'
        } finally {
            if ($null -eq $origPwsh) { [Environment]::SetEnvironmentVariable('BONSAI_PUSH_PWSH_PATH', [NullString]::Value) } else { [Environment]::SetEnvironmentVariable('BONSAI_PUSH_PWSH_PATH', $origPwsh) }
            Remove-CiScratch $scratch
        }
    }
}

try {
    if (-not $ReportPath -and @($Case).Count -le 1) {
        $selectedCase = if ($Case -and $Case.Count -eq 1) { $Case[0] } else { '' }
        Invoke-CiCases $cases $selectedCase
    } else {
        [string[]]$selected = if ($Case -and @($Case).Count -gt 0) {
            foreach ($name in $Case) {
                if (-not $cases.Contains($name)) { throw "Unknown named case: $name" }
            }
            @($Case)
        } else {
            @($cases.Keys)
        }
        $results = [ordered]@{}
        $failed = $false
        foreach ($name in $selected) {
            try {
                & $cases[$name]
                $results[$name] = @{ passed = $true; error = $null }
                "PASS $name"
            } catch {
                $failed = $true
                $results[$name] = @{ passed = $false; error = $_.Exception.Message }
                [Console]::Error.WriteLine("FAIL test_push_sanity.ps1::${name}: $($_.Exception.Message)")
            }
        }
        if ($ReportPath) {
            $results | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $ReportPath -Encoding utf8NoBOM
        }
        if ($failed) { throw 'One or more push sanity fixtures failed.' }
        "$(@($selected).Count) focused fixture(s) passed."
    }
} finally {
    foreach ($entry in $script:PushTemplateCache.Values) {
        Remove-CiScratch $entry.Scratch
    }
}
