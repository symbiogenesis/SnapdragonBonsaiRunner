#Requires -Version 7.6
param([string]$Case,[string]$CandidateRoot)
. (Join-Path $PSScriptRoot 'test_support.ps1')
$repository = if ($CandidateRoot) { (Resolve-Path -LiteralPath $CandidateRoot).ProviderPath } else { Split-Path (Split-Path $PSScriptRoot -Parent) -Parent }
$action = [IO.File]::ReadAllText((Join-Path $repository '.github/actions/check-rust/action.yml'))
$steps = Get-CiActionSteps $action
$execution = @($steps.Values | Where-Object { $_ -cmatch '(?m)^      id: rust-check\s*$' })
Assert-CiEqual $execution.Count 1
$body = [scriptblock]::Create((Get-CiActionBody $execution[0]))

function Invoke-IncrementalFixtureTool {
    $script:observation = @{
        Arguments = @($args)
        Incremental = $env:CARGO_INCREMENTAL
        ReleaseIncremental = $env:CARGO_PROFILE_RELEASE_INCREMENTAL
        CodegenUnits = $env:CARGO_PROFILE_RELEASE_CODEGEN_UNITS
    }
}

function Invoke-IncrementalFixture([string]$Incremental, [string]$Release = 'false', [string]$Environment = 'github-hosted') {
    $scratch = New-CiScratch 'bonsai-incremental-ci-'
    $values = @{
        INCREMENTAL = $Incremental
        GITHUB_RELEASE_BUILD = $Release
        RUNNER_ENVIRONMENT = $Environment
        BONSAI_CI_PATH = 'Invoke-IncrementalFixtureTool'
        BROKER_PACKAGES = '["bonsai-tools"]'
        GITHUB_WORKSPACE = $repository
        CARGO_TARGET_DIR = (Join-Path $scratch 'target')
        GITHUB_ENV = (Join-Path $scratch 'environment')
        RUNNER_TEMP = $scratch
        CARGO_INCREMENTAL = '0'
        CARGO_PROFILE_RELEASE_INCREMENTAL = 'false'
        CARGO_PROFILE_RELEASE_CODEGEN_UNITS = '7'
        BUILD_ONLY = 'false'
    }
    $original = @{}
    foreach ($name in $values.Keys) { $original[$name] = [Environment]::GetEnvironmentVariable($name) }
    try {
        foreach ($name in $values.Keys) { [Environment]::SetEnvironmentVariable($name, $values[$name]) }
        $script:observation = $null
        & $body | Out-Null
        Assert-Ci ($null -ne $script:observation) 'The selected checks never reached the native executor.'
        return $script:observation
    } finally {
        foreach ($name in $original.Keys) {
            $value = if ($null -eq $original[$name]) { [NullString]::Value } else { $original[$name] }
            [Environment]::SetEnvironmentVariable($name, $value)
        }
        Remove-CiScratch $scratch
    }
}

$cases = [ordered]@{
    incremental_preserves_release_policy = {
        $result = Invoke-IncrementalFixture 'true'
        Assert-CiEqual $result.Incremental '1'
        Assert-CiEqual $result.ReleaseIncremental 'true'
        Assert-CiEqual $result.CodegenUnits '16'
        Assert-CiEqual $result.Arguments[0] 'rust-check'
        foreach ($required in @('--full-tests', '--build', '--packages-json')) {
            Assert-Ci ($required -cin $result.Arguments) "Incremental mode removed $required."
        }
        Assert-Ci ('--github-release' -cnotin $result.Arguments)
    }
    disabled_preserves_ambient_settings_and_publication = {
        foreach ($release in @('false', 'true')) {
            $result = Invoke-IncrementalFixture 'false' $release
            Assert-CiEqual $result.Incremental '0'
            Assert-CiEqual $result.ReleaseIncremental 'false'
            Assert-CiEqual $result.CodegenUnits '7'
            Assert-CiEqual ('--github-release' -cin $result.Arguments) ($release -ceq 'true')
        }
    }
    invalid_or_unapproved_modes_fail_before_execution = {
        foreach ($value in @('', 'TRUE', '1', "true`n")) {
            Assert-CiReject { Invoke-IncrementalFixture $value }
            Assert-Ci ($null -eq $script:observation) 'Invalid incremental input reached the native executor.'
        }
        Assert-CiReject { Invoke-IncrementalFixture 'true' 'true' }
        Assert-Ci ($null -eq $script:observation) 'Incremental publication reached the native executor.'
        Assert-CiReject { Invoke-IncrementalFixture 'true' 'false' 'self-hosted' }
        Assert-Ci ($null -eq $script:observation) 'Owner-device incremental mode reached the native executor.'
    }
}
Invoke-CiCases $cases $Case
