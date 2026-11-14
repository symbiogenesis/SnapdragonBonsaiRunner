#Requires -Version 7.6
param([Parameter(Mandatory)][string]$ActionlintPath, [string]$Case)
. (Join-Path $PSScriptRoot 'test_support.ps1')
$checker = Join-Path $PSScriptRoot 'check_workflow_syntax.ps1'
function Invoke-ArchiveCacheFixture([scriptblock]$Work) {
    $scratch = New-CiScratch 'bonsai-actionlint-cache-'
    try {
        . $checker
        & $Work $scratch
    } finally { Remove-CiScratch $scratch }
}
function Invoke-SyntaxFixture([string]$Source, [bool]$Reject = $false, [bool]$WithoutGit = $false) {
    $scratch = New-CiScratch 'bonsai-workflow-syntax-'
    try {
        if (-not $WithoutGit) { [void][IO.Directory]::CreateDirectory((Join-Path $scratch '.git')) }
        if ($WithoutGit) { Write-CiFixture $scratch '.github/actionlint.yaml' "self-hosted-runner:`n  labels:`n    - fixture-snapdragon`n" }
        Write-CiFixture $scratch '.github/workflows/fixture.yml' $Source
        Write-CiFixture $scratch '.github/actions/fixture/action.yml' @'
name: Fixture
description: Inert input contract
inputs:
  known:
    required: false
    description: Known input
runs:
  using: composite
  steps:
    - shell: pwsh
      run: Write-Output fixture
'@
        if ($Reject) { Assert-CiReject { & $checker -Root $scratch -ActionlintPath $ActionlintPath *> $null } }
        else { & $checker -Root $scratch -ActionlintPath $ActionlintPath }
    } finally { Remove-CiScratch $scratch }
}
$valid = @'
name: Fixture
on: workflow_dispatch
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: $/.github/actions/fixture
        with:
          known: value
      - shell: pwsh
        run: |
          # The linter must leave script text alone.
          Write-Output 'uses: $/literal-text'
'@
$cases = [ordered]@{
    archive_cache_reuses_verified_bytes_and_rejects_poison = { Invoke-ArchiveCacheFixture { param($scratch)
        $payload = Join-Path $scratch 'payload.zip'
        [IO.File]::WriteAllText($payload, 'Inert archive bytes; never extracted or executed.')
        $digest = (Get-FileHash -LiteralPath $payload -Algorithm SHA256).Hash.ToLowerInvariant()
        $downloads = [Collections.Generic.List[string]]::new()
        function Invoke-WebRequest {
            param($Uri, $OutFile, $ConnectionTimeoutSeconds, $OperationTimeoutSeconds, $MaximumRetryCount, $RetryIntervalSec)
            $downloads.Add($Uri)
            [IO.File]::Copy($payload, $OutFile, $false)
        }
        $cache = Join-Path $scratch 'cache'
        $first = Join-Path $scratch 'first.zip'; $second = Join-Path $scratch 'second.zip'; $third = Join-Path $scratch 'third.zip'
        Assert-CiEqual (Get-PinnedActionlintArchive 'https://fixture.invalid/pinned.zip' $digest 'asset.zip' $first $cache) $first
        Assert-CiEqual $downloads.Count 1
        Assert-CiEqual (Get-PinnedActionlintArchive 'https://fixture.invalid/pinned.zip' $digest 'asset.zip' $second $cache) $second
        Assert-CiEqual $downloads.Count 1
        Assert-CiEqual ([IO.File]::ReadAllText($first)) ([IO.File]::ReadAllText($second))
        $cached = Join-Path (Join-Path $cache $digest) 'asset.zip'
        [IO.File]::WriteAllText($cached, 'Poisoned cached bytes must never be extracted.')
        Assert-CiReject { Get-PinnedActionlintArchive 'https://fixture.invalid/pinned.zip' $digest 'asset.zip' $third $cache } 'Pinned actionlint archive checksum differs'
        Assert-CiEqual $downloads.Count 1
        Assert-CiEqual ([IO.File]::ReadAllText($cached)) 'Poisoned cached bytes must never be extracted.'
    } }
    archive_cache_rejects_download_before_publication = { Invoke-ArchiveCacheFixture { param($scratch)
        $payload = Join-Path $scratch 'payload.zip'
        [IO.File]::WriteAllText($payload, 'Expected inert bytes.')
        $digest = (Get-FileHash -LiteralPath $payload -Algorithm SHA256).Hash.ToLowerInvariant()
        function Invoke-WebRequest {
            param($Uri, $OutFile, $ConnectionTimeoutSeconds, $OperationTimeoutSeconds, $MaximumRetryCount, $RetryIntervalSec)
            [IO.File]::WriteAllText($OutFile, 'Rejected download bytes.')
        }
        $cache = Join-Path $scratch 'cache'
        Assert-CiReject { Get-PinnedActionlintArchive 'https://fixture.invalid/pinned.zip' $digest 'asset.zip' (Join-Path $scratch 'download.zip') $cache } 'Pinned actionlint archive checksum differs'
        Assert-Ci (-not (Test-Path -LiteralPath $cache)) 'A rejected download published cache data.'
    } }
    snapshot_custom_runner_labels_use_explicit_configuration = {
        Invoke-SyntaxFixture ($valid.Replace('runs-on: ubuntu-latest', 'runs-on: [self-hosted, fixture-snapdragon]')) $false $true
    }
    same_repository_action_and_script_text_pass = { Invoke-SyntaxFixture $valid }
    quoted_same_repository_action_keys_pass = {
        Invoke-SyntaxFixture ($valid.Replace('uses: $/', "'uses': `$/"))
        Invoke-SyntaxFixture ($valid.Replace('uses: $/', '"uses": $/'))
    }
    syntax_rejects_unknown_job_dependency = { Invoke-SyntaxFixture ($valid.Replace('    runs-on:', "    needs: missing`n    runs-on:")) $true }
    syntax_rejects_invalid_expression = { Invoke-SyntaxFixture ($valid.Replace('known: value', 'known: ${{ github. }}')) $true }
    syntax_checks_normalized_local_action_inputs = {
        Invoke-SyntaxFixture $valid
        Invoke-SyntaxFixture ($valid.Replace('known: value', 'unknown: value')) $true
    }
}
Invoke-CiCases $cases $Case
