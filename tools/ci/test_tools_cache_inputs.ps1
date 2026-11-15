#Requires -Version 7.6
param(
    [string]$Case,
    [string]$CandidateRoot,
    [ValidateSet('bonsai-tools','ci-tool')][string]$EnvironmentAction='bonsai-tools'
)
. (Join-Path $PSScriptRoot 'test_support.ps1')
$repository=if ($CandidateRoot) { (Resolve-Path -LiteralPath $CandidateRoot).ProviderPath } else { Split-Path (Split-Path $PSScriptRoot -Parent) -Parent }
$actionPath=Join-Path $repository '.github/actions/bonsai-tools/action.yml'
function Restore-CacheEnvironment([string]$Name,$Value) {
    # PowerShell 7.5+ converts $null to an empty string for this .NET method.
    # Preserve real empty originals while removing variables originally absent.
    if ($null -eq $Value) { [Environment]::SetEnvironmentVariable($Name,[NullString]::Value) }
    else { [Environment]::SetEnvironmentVariable($Name,$Value) }
}
# Cargo's checked-in manifests use one-line path/workspace declarations. Fail
# closed for unfamiliar local declaration syntax rather than guessing a closure.
function Get-CargoDependencyDeclarations([string]$Manifest,[bool]$WorkspaceOnly=$false) {
    $table=''; $result=@{}
    foreach ($line in [IO.File]::ReadAllLines($Manifest)) {
        if ($line -cmatch '^\[([^\[].*)\]\s*$') { $table=$Matches[1]; continue }
        if ($line -cmatch '^\[\[') { $table=''; continue }
        $selected=if ($WorkspaceOnly) { $table -ceq 'workspace.dependencies' } else { $table -cmatch '^(?:(?:target\..+)\.)?(?:build-|dev-)?dependencies$' }
        if (-not $selected -or $line -cmatch '^\s*(?:#|$)') { continue }
        if ($line -cnotmatch '^\s*([A-Za-z0-9_-]+)(\.workspace)?\s*=\s*(.+?)\s*$') { throw "Extend cache closure controls for dependency syntax in $Manifest" }
        $name=$Matches[1]; $workspace=$Matches[2]; $value=$Matches[3]
        if ($workspace) { if ($value -cne 'true') { throw 'Unexpected workspace dependency declaration' }; $result[$name]=@{Workspace=$true}; continue }
        if ($value -cmatch '\bpath\s*=\s*"([^"\\]+)"') { $result[$name]=@{Path=$Matches[1]}; continue }
        if ($value -cmatch '\bworkspace\s*=\s*true\b') { $result[$name]=@{Workspace=$true}; continue }
        if ($value -cmatch '\b(?:path|workspace)\b' -or ($value.StartsWith('{') -and -not $value.EndsWith('}'))) { throw "Extend cache closure controls for local dependency syntax in $Manifest" }
        $result[$name]=@{}
    }
    return $result
}
function Get-CargoPackageClosure([string]$Root,[string]$Initial) {
    $Root=[IO.Path]::GetFullPath($Root); $workspace=Get-CargoDependencyDeclarations (Join-Path $Root 'Cargo.toml') $true
    $pending=[Collections.Generic.Stack[string]]::new(); $pending.Push((Join-Path $Root $Initial)); $result=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    while ($pending.Count) {
        $package=[IO.Path]::GetFullPath($pending.Pop()); if (-not $package.StartsWith($Root+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Path dependency escaped repository' }; if (-not $result.Add($package)) { continue }
        $declarations=Get-CargoDependencyDeclarations (Join-Path $package 'Cargo.toml')
        foreach ($name in $declarations.Keys) {
            $declaration=$declarations[$name]; $base=$package
            if ($declaration.ContainsKey('Workspace')) { if (-not $workspace.ContainsKey($name)) { throw "Unknown workspace dependency $name" }; $declaration=$workspace[$name]; $base=$Root }
            if ($declaration.ContainsKey('Path')) { $pending.Push((Join-Path $base $declaration.Path)) }
        }
    }
    return ,@($result)
}
function Get-CargoEmbeddedFiles([string]$Package,[string]$Source) {
    $text=([IO.File]::ReadAllText($Source) -split '#\[cfg\(test\)\]',2)[0]
    $ordinary=[regex]::Matches($text,'include_(?:str|bytes)!\(\s*"([^"\\]+)"\s*\)')
    $manifestRelative=[regex]::Matches($text,'include_(?:str|bytes)!\(\s*concat!\(\s*env!\("CARGO_MANIFEST_DIR"\),\s*"([^"\\]+)"\s*\)\s*\)')
    if ([regex]::Matches($text,'\binclude_(?:str|bytes)!\s*\(').Count -ne $ordinary.Count+$manifestRelative.Count) { throw "Extend cache closure controls for an unfamiliar include expression in $Source" }
    foreach ($match in $ordinary) { [IO.Path]::GetFullPath((Join-Path ([IO.Path]::GetDirectoryName($Source)) $match.Groups[1].Value)) }
    foreach ($match in $manifestRelative) { [IO.Path]::GetFullPath((Join-Path $Package $match.Groups[1].Value.TrimStart('/'))) }
    if ([IO.Path]::GetFileName($Source) -ceq 'build.rs') { foreach ($match in [regex]::Matches($text,'\.set_icon\("([^"\\]+)"\)')) { [IO.Path]::GetFullPath((Join-Path $Package $match.Groups[1].Value)) } }
}
function Get-RequiredToolsFiles([string]$Root,[string[]]$Packages=@('tools/bonsai')) {
    $files=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($name in @('Cargo.toml','Cargo.lock','rust-toolchain.toml')) { [void]$files.Add((Join-Path $Root $name)) }
    foreach ($file in Get-ChildItem -LiteralPath (Join-Path $Root '.cargo') -File -Recurse) { [void]$files.Add($file.FullName) }
    $closure=@($Packages | ForEach-Object { Get-CargoPackageClosure $Root $_ } | ForEach-Object { $_ } | Sort-Object -Unique)
    if (-not (Get-Variable -Name requiredPackageFilesCache -Scope Script -ErrorAction SilentlyContinue)) { $script:requiredPackageFilesCache = @{} }
    foreach ($package in $closure) {
        $cacheKey = "$Root`n$package"
        $packageFiles = $script:requiredPackageFilesCache[$cacheKey]
        if ($null -eq $packageFiles) {
            $packageFiles = [Collections.Generic.List[string]]::new()
            $packageFiles.Add((Join-Path $package 'Cargo.toml')); $sources=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
            foreach ($file in Get-ChildItem -LiteralPath $package -File -Filter '*.rs' -Recurse) { [void]$sources.Add($file.FullName) }
            # Binaries may explicitly point outside the package directory.
            foreach ($match in [regex]::Matches([IO.File]::ReadAllText((Join-Path $package 'Cargo.toml')),'(?m)^path\s*=\s*"([^"\\]+\.rs)"\s*$')) { [void]$sources.Add([IO.Path]::GetFullPath((Join-Path $package $match.Groups[1].Value))) }
            foreach ($source in $sources) {
                $packageFiles.Add($source); $relative=[IO.Path]::GetRelativePath($package,$source).Replace('\','/')
                if ('tests' -cin ($relative -split '/') -or $source -cmatch '(?:_tests|tests)\.rs$') { continue }
                foreach ($embedded in @(Get-CargoEmbeddedFiles $package $source)) { $packageFiles.Add($embedded) }
            }
            $script:requiredPackageFilesCache[$cacheKey] = $packageFiles
        }
        foreach ($entry in $packageFiles) { [void]$files.Add($entry) }
    }
    foreach ($file in $files) { $relative=[IO.Path]::GetRelativePath($Root,[IO.Path]::GetFullPath($file)).Replace('\','/'); if ($relative.StartsWith('../',[StringComparison]::Ordinal)) { throw 'Embedded build input escaped repository' }; $relative }
}
function Get-ToolsHashPatterns {
    $text=[IO.File]::ReadAllText($actionPath)
    if ($text -cnotmatch 'TOOLS_SOURCE_ID: \$\{\{ hashFiles\((.*)\) \}\}') { throw 'Missing exact packaged-tool source hash.' }
    return @([regex]::Matches($Matches[1],'''([^'']+)''') | ForEach-Object { $_.Groups[1].Value })
}
function Get-CiToolHashPatterns {
    $text=[IO.File]::ReadAllText((Join-Path $repository '.github/actions/ci-tool/action.yml'))
    if ($text -cnotmatch 'CI_SOURCE_ID: \$\{\{ hashFiles\((.*)\) \}\}') { throw 'Missing exact CI-tool source hash.' }
    return @([regex]::Matches($Matches[1],'''([^'']+)''') | ForEach-Object { $_.Groups[1].Value })
}
function Get-UncoveredToolsFiles([string[]]$Files,[string[]]$Patterns) {
    $compiled = @($Patterns | ForEach-Object { [Management.Automation.WildcardPattern]::new($_,[Management.Automation.WildcardOptions]::CultureInvariant) })
    foreach ($file in $Files) { $covered=$false; foreach ($wildcard in $compiled) { if ($wildcard.IsMatch($file)) { $covered=$true; break } }; if (-not $covered) { $file } }
}
function Get-LaneCachePatterns([string]$Lane) {
    $action=[IO.File]::ReadAllText((Join-Path $repository '.github/actions/check-rust/action.yml')).Replace("`r`n","`n")
    $key=([regex]::Match($action,'(?m)^        key: (.+)$')).Groups[1].Value
    $branches=@([regex]::Matches($key,'inputs\.lane == ''([^'']+)'' && hashFiles\(([^)]+)\)'))
    Assert-Ci ($branches.Count -gt 0) 'Expected lane-specific cache identities.'
    $common=$key.Substring(0,$branches[0].Index)
    $selected=@($branches | Where-Object { $_.Groups[1].Value -ceq $Lane })
    $source=if ($selected.Count) { $selected[0].Groups[2].Value } else { ([regex]::Match($key,'\|\| hashFiles\(([^)]+)\)')).Groups[1].Value }
    Assert-Ci (-not [string]::IsNullOrEmpty($source)) "Missing source identity for $Lane"
    foreach ($hash in [regex]::Matches($common,'hashFiles\(([^)]+)\)')) { foreach ($match in [regex]::Matches($hash.Groups[1].Value,'''([^'']+)''')) { $match.Groups[1].Value } }
    foreach ($match in [regex]::Matches($source,'''([^'']+)''')) { $match.Groups[1].Value }
}
function Invoke-LanePhaseFixture([string]$BuildOnly) {
    $scratch=New-CiScratch 'bonsai-lane-phase-'; $original=@{}
    $environment=@{BUILD_ONLY=$BuildOnly;INCREMENTAL='false';GITHUB_RELEASE_BUILD='false';BONSAI_CI_PATH='Invoke-CapturedLane';RUNNER_TEMP=$scratch;GITHUB_ENV=(Join-Path $scratch 'environment');GITHUB_WORKSPACE=$repository;BROKER_PACKAGES='["bonsai-tools"]';CARGO_TARGET_DIR=(Join-Path $scratch 'target')}
    foreach ($key in $environment.Keys) { $original[$key]=[Environment]::GetEnvironmentVariable($key) }
    try {
        foreach ($key in $environment.Keys) { [Environment]::SetEnvironmentVariable($key,$environment[$key]) }
        function Invoke-CapturedLane { $script:capturedLaneArguments=@($args) }
        $steps=Get-CiActionSteps ([IO.File]::ReadAllText((Join-Path $repository '.github/actions/check-rust/action.yml')))
        & ([scriptblock]::Create((Get-CiActionBody $steps['Format, analyze, document, test and build the selected crates']))) | Out-Null
        return ,$script:capturedLaneArguments
    } finally { foreach ($key in $original.Keys) { Restore-CacheEnvironment $key $original[$key] }; Remove-CiScratch $scratch }
}
function Invoke-ToolsEnvironmentFixture([object[]]$Scenarios) {
    # Linux can contain both RUSTFLAGS and RustFlags. Preserve the actual names
    # so fixture cleanup does not merge distinct ambient environment entries.
    $scratch=New-CiScratch 'bonsai-tools-identity-'; $original=[hashtable]::new([StringComparer]::Ordinal); foreach ($entry in [Environment]::GetEnvironmentVariables().GetEnumerator()) { $original[$entry.Key]=$entry.Value }
    try {
        foreach ($name in @([Environment]::GetEnvironmentVariables().Keys)) { if ($name.ToUpperInvariant().StartsWith('RUST',[StringComparison]::Ordinal) -or $name.ToUpperInvariant().StartsWith('CARGO_',[StringComparison]::Ordinal)) { [Environment]::SetEnvironmentVariable($name,[NullString]::Value) } }
        $cargoDirectory=Join-Path $scratch 'cargo'; [void][IO.Directory]::CreateDirectory($cargoDirectory)
        $environment=@{TOOLS_SOURCE_ID=('a'*64);CI_SOURCE_ID=('a'*64);RUNNER_ENVIRONMENT='github-hosted';ImageOS='fixture-os';ImageVersion='fixture-image';RUNNER_TEMP=$scratch;CARGO_HOME=$cargoDirectory;GITHUB_ENV=(Join-Path $scratch 'environment')}; foreach ($key in $environment.Keys) { [Environment]::SetEnvironmentVariable($key,$environment[$key]) }
        # Keep this harness and its assertions in the committed checking policy;
        # only the selected immutable candidate's adapter supplies the body.
        $prefix=if ($EnvironmentAction -ceq 'ci-tool') { 'BONSAI_CI' } else { 'BONSAI_TOOLS' }
        $action=[IO.File]::ReadAllText((Join-Path $repository ".github/actions/$EnvironmentAction/action.yml")); $begin=$action.IndexOf('        $buildInputs = '); if ($begin -lt 0) { throw 'Missing candidate cache-input adapter.' }; $end=$action.IndexOf(('        "'+$prefix+'_SOURCE_ID='),$begin); if ($end -le $begin) { throw 'Missing candidate cache-input output boundary.' }; $script=(($action.Substring($begin,$end-$begin) -split '\r?\n' | ForEach-Object { if ($_.Length -ge 8) { $_.Substring(8) } else { '' } }) -join "`n")
        $result=@()
        foreach ($scenario in $Scenarios) {
            # NullString passes an actual null through PowerShell's string
            # parameter binder; ordinary $null would leave an empty override.
            foreach ($key in $scenario.environment.Keys) { if ($null -eq $scenario.environment[$key]) { [Environment]::SetEnvironmentVariable($key,[NullString]::Value) } }
            foreach ($key in $scenario.environment.Keys) { if ($null -ne $scenario.environment[$key]) { [Environment]::SetEnvironmentVariable($key,$scenario.environment[$key]) } }
            foreach ($name in @('config','config.toml')) { $path=Join-Path $cargoDirectory $name; if ($scenario.ContainsKey($name)) { [IO.File]::WriteAllText($path,$scenario[$name]) } elseif ([IO.File]::Exists($path)) { [IO.File]::Delete($path) } }
            [IO.File]::WriteAllText($env:GITHUB_ENV,''); & ([scriptblock]::Create($script)) | Out-Null
            $identity=@([IO.File]::ReadAllLines($env:GITHUB_ENV) | Where-Object { $_.StartsWith(($prefix+'_CACHE_HOST='),[StringComparison]::Ordinal) })
            Assert-CiEqual $identity.Count 1
            $result+=$identity
        }
        return ,$result
    } finally { foreach ($name in @([Environment]::GetEnvironmentVariables().Keys)) { if (-not $original.ContainsKey($name)) { [Environment]::SetEnvironmentVariable($name,[NullString]::Value) } }; foreach ($name in $original.Keys) { Restore-CacheEnvironment $name $original[$name] }; Remove-CiScratch $scratch }
}
$cases=[ordered]@{
 lane_dependency_and_embedded_closures_hashed={
    $lanes=@{ui=@('src/chat-ui');core=@('src/broker','src/speech-service');asr=@('src/speech-asr');tts=@('src/speech-tts');tools=@('tools/bonsai');maintenance=@('tools/bonsai','tools/ci','src/audio','src/artifacts','src/cpu-policy','src/windows-process','src/windows-credentials')}
    foreach ($lane in $lanes.Keys) { $files=@(Get-RequiredToolsFiles $repository $lanes[$lane]); Assert-CiEqual @(Get-UncoveredToolsFiles $files @(Get-LaneCachePatterns $lane)) @() }
 }
 lane_audio_and_external_source_omissions_detected={
    foreach ($lane in @('ui','core','asr','tts','tools','maintenance')) {
        $patterns=@(Get-LaneCachePatterns $lane); $path='src/audio/src/lib.rs'
        Assert-CiEqual @(Get-UncoveredToolsFiles @($path) $patterns) @()
        $reduced=@($patterns | Where-Object { -not [Management.Automation.WildcardPattern]::new($_).IsMatch($path) })
        Assert-CiEqual @(Get-UncoveredToolsFiles @($path) $reduced) @($path)
    }
    foreach ($path in @('.cargo/config.toml','tools/native-reference/reference_numerics.rs','src/chat-ui/assets/bonsai.ico','LICENSE','NOTICE')) { Assert-CiEqual @(Get-UncoveredToolsFiles @($path) @(Get-LaneCachePatterns 'tools')) @() }
 }
 lane_producer_and_checks_save_distinct_snapshots={
    $action=[IO.File]::ReadAllText((Join-Path $repository '.github/actions/check-rust/action.yml')).Replace("`r`n","`n")
    $key=([regex]::Match($action,'(?m)^        key: (.+)$')).Groups[1].Value
    $suffix='${{ inputs.build-only == ''true'' && ''build'' || ''checks'' }}'
    Assert-Ci ($key.EndsWith($suffix,[StringComparison]::Ordinal))
    $sameSource=$key.Substring(0,$key.Length-$suffix.Length)
    $firstRestore=([regex]::Match($action,'restore-keys: \|\r?\n          (.+)')).Groups[1].Value
    Assert-CiEqual $firstRestore $sameSource
    Assert-Ci ($action.Contains("steps.cargo-cache.outputs.cache-hit != 'true'"))
    Assert-Ci ($action.Contains('name: broker-checks-${{ inputs.lane }}-${{ github.job }}-${{ github.run_id }}'))
 }
 lane_build_only_never_replaces_default_full_checks={
    $build=Invoke-LanePhaseFixture 'true'; Assert-Ci ('--phase' -cin $build); Assert-Ci ('build' -cin $build); Assert-Ci ('--full-tests' -cnotin $build)
    foreach ($mode in @('false','')) { $checks=Invoke-LanePhaseFixture $mode; Assert-Ci ('--full-tests' -cin $checks); Assert-Ci ('--build' -cin $checks); Assert-Ci ('--phase' -cnotin $checks) }
 }
 real_dependency_embedded_closure_hashed={ $files=@(Get-RequiredToolsFiles $repository); Assert-CiEqual @(Get-UncoveredToolsFiles $files (Get-ToolsHashPatterns)) @(); foreach ($file in $files) { Assert-Ci (Test-Path -LiteralPath (Join-Path $repository $file) -PathType Leaf) $file } }
 ci_tool_dependency_and_embedded_closure_hashed={ $files=@(Get-RequiredToolsFiles $repository @('tools/ci')); Assert-CiEqual @(Get-UncoveredToolsFiles $files (Get-CiToolHashPatterns)) @(); foreach ($file in $files) { Assert-Ci (Test-Path -LiteralPath (Join-Path $repository $file) -PathType Leaf) $file } }
 missing_audio_and_embedded_input_detected={ $files=@(Get-RequiredToolsFiles $repository); $patterns=Get-ToolsHashPatterns; foreach ($path in @('src/audio/src/lib.rs','src/speech-asr/model.lock.json','src/speech-tts/model.lock.json','tools/qwen-tts/model.lock.json','tools/hexagon/diagnostic.manifest','src/chat-ui/assets/bonsai.ico','LICENSE','NOTICE')) { Assert-Ci ($path -cin $files) $path; $reduced=@($patterns | Where-Object { -not [Management.Automation.WildcardPattern]::new($_).IsMatch($path) }); Assert-Ci ($path -cin @(Get-UncoveredToolsFiles $files $reduced)) $path } }
 traversal_and_literals_use_manifest_owner={
    $scratch=New-CiScratch 'bonsai-cache-closure-'
    try {
        Write-CiFixture $scratch 'Cargo.toml' "[workspace.dependencies]`nshared = { path = `"shared`" }`n"
        Write-CiFixture $scratch 'tools/Cargo.toml' "[dependencies]`nshared.workspace = true`n[target.'cfg(windows)'.build-dependencies]`nleaf = { path = `"../leaf`" }`n"
        Write-CiFixture $scratch 'shared/Cargo.toml' ''; Write-CiFixture $scratch 'leaf/Cargo.toml' ''
        Assert-CiSet (Get-CargoPackageClosure $scratch 'tools') @((Join-Path $scratch 'tools'),(Join-Path $scratch 'shared'),(Join-Path $scratch 'leaf'))
        $source=Join-Path $scratch 'tools/build.rs'; [IO.File]::WriteAllText($source,'const A: &str = include_str!("../shared/data");'+"`n"+'const B: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../leaf/data"));'+"`n"+'r.set_icon("../leaf/icon.ico");')
        Assert-CiSet @(Get-CargoEmbeddedFiles (Split-Path $source -Parent) $source) @((Join-Path $scratch 'shared/data'),(Join-Path $scratch 'leaf/data'),(Join-Path $scratch 'leaf/icon.ico'))
        [IO.File]::WriteAllText($source,'const A: &str = include_str!(unreviewed!());'); Assert-CiReject { Get-CargoEmbeddedFiles (Split-Path $source -Parent) $source }
    } finally { Remove-CiScratch $scratch }
 }
 native_caches_share_environment_filter={ $filters=@(); foreach ($name in @('bonsai-tools','ci-tool')) { $text=[IO.File]::ReadAllText((Join-Path $repository ".github/actions/$name/action.yml")); $filters+=,@($text -split '\r?\n' | Where-Object { $_.Contains('$buildInputs = @(') } | ForEach-Object Trim) }; Assert-CiEqual $filters[0].Count 1; Assert-CiEqual $filters[0] $filters[1] }
 both_cache_layers_bind_same_inputs={ $action=[IO.File]::ReadAllText($actionPath); $exact=(($action -split '    - name: Reuse only the exact packaged CLI',2)[1] -split '    - name:',2)[0]; $incremental=(($action -split '    - name: Reuse Cargo downloads',2)[1] -split '    - uses:',2)[0]; foreach ($block in @($exact,$incremental)) { Assert-Ci ($block.Contains('${{ env.BONSAI_TOOLS_SOURCE_ID }}')); Assert-Ci ($block.Contains('${{ env.BONSAI_TOOLS_CACHE_HOST }}')) }; Assert-Ci (-not $exact.Contains('restore-keys:')); Assert-Ci ($action.Contains('HostId = $env:BONSAI_TOOLS_CACHE_HOST')); Assert-Ci ($action.Contains('$receipt.HostId -cne $env:BONSAI_TOOLS_CACHE_HOST')) }
 compiler_profile_and_cargo_config_changes_invalidate={
    foreach ($variable in @('RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','RUSTC','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER','RUSTC_BOOTSTRAP','CARGO_INCREMENTAL','CARGO_BUILD_RUSTFLAGS','CARGO_PROFILE_RELEASE_LTO','CARGO_TARGET_AARCH64_PC_WINDOWS_MSVC_LINKER')) { $values=Invoke-ToolsEnvironmentFixture @(@{environment=@{}},@{environment=@{$variable='changed'}}); Assert-Ci ($values[0] -cne $values[1]) $variable }
    foreach ($filename in @('config','config.toml')) { $values=Invoke-ToolsEnvironmentFixture @(@{environment=@{}},@{environment=@{};$filename="[build]`njobs = 2`n"},@{environment=@{};$filename="[build]`njobs = 4`n"}); Assert-CiEqual @($values | Sort-Object -Unique).Count 3 }
 }
 mixed_case_names_have_same_build_identity={
    foreach ($pair in @(@('RUSTFLAGS','RustFlags'),@('CARGO_PROFILE_RELEASE_LTO','Cargo_Profile_Release_Lto'))) {
        # Hash literals reject case variants even on Linux. The transition must
        # retain both operations: remove the old spelling, then set the new one.
        $transition=[hashtable]::new([StringComparer]::Ordinal); $transition[$pair[0]]=$null; $transition[$pair[1]]='changed'
        $values=Invoke-ToolsEnvironmentFixture @(@{environment=@{}},@{environment=@{$pair[0]='changed'}},@{environment=$transition},@{environment=@{$pair[1]=$null}})
        Assert-CiEqual $values[0] $values[3]; Assert-CiEqual $values[1] $values[2]; Assert-Ci ($values[0] -cne $values[1])
    }
 }
 environment_fixture_preserves_absent_and_empty_values={
    $names=@('CARGO_BUILD_CACHE_FIXTURE_ABSENT','CARGO_BUILD_CACHE_FIXTURE_EMPTY'); $before=[hashtable]::new([StringComparer]::Ordinal)
    foreach ($name in $names) { $before[$name]=[Environment]::GetEnvironmentVariable($name) }
    try {
        [Environment]::SetEnvironmentVariable($names[0],[NullString]::Value)
        [Environment]::SetEnvironmentVariable($names[1],'')
        $values=Invoke-ToolsEnvironmentFixture @(@{environment=@{}},@{environment=@{RUSTFLAGS=''}},@{environment=@{RUSTFLAGS=$null}})
        Assert-CiEqual $values[0] $values[2]
        Assert-Ci ($values[0] -cne $values[1]) 'An explicit empty override must remain distinct from an absent override.'
        Assert-Ci ($null -eq [Environment]::GetEnvironmentVariable($names[0])) 'Fixture cleanup created an override that was originally absent.'
        Assert-CiEqual ([Environment]::GetEnvironmentVariable($names[1])) ''
    } finally { foreach ($name in $names) { Restore-CacheEnvironment $name $before[$name] } }
 }
 environment_fixture_restores_case_sensitive_names={
    $names=@('CARGO_BUILD_CACHE_FIXTURE','Cargo_Build_Cache_Fixture'); $before=[hashtable]::new([StringComparer]::Ordinal)
    foreach ($name in $names) { $before[$name]=[Environment]::GetEnvironmentVariable($name) }
    try {
        [Environment]::SetEnvironmentVariable($names[0],'uppercase')
        if (-not $IsWindows) { [Environment]::SetEnvironmentVariable($names[1],'mixed-case') }
        Invoke-ToolsEnvironmentFixture @(@{environment=@{}}) | Out-Null
        Assert-CiEqual ([Environment]::GetEnvironmentVariable($names[0])) 'uppercase'
        if (-not $IsWindows) { Assert-CiEqual ([Environment]::GetEnvironmentVariable($names[1])) 'mixed-case' }
    } finally { foreach ($name in $names) { Restore-CacheEnvironment $name $before[$name] } }
 }
 metadata_and_override_paths_do_not_partition={ $values=Invoke-ToolsEnvironmentFixture @(@{environment=@{}},@{environment=@{GITHUB_RUN_ID='new-run';CARGO_TARGET_DIR='different-target';CARGO_BUILD_FINGERPRINT='mtime'}}); Assert-CiEqual $values[0] $values[1] }
}
Invoke-CiCases $cases $Case
