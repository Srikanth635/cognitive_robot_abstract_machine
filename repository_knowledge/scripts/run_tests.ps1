# %% test selection

param(
    [string]$Repository = (Resolve-Path "$PSScriptRoot/../..").Path,
    [string[]]$TestPaths = @("test/repository_knowledge_test"),
    [long]$MinimumFreeBytes = 8GB,
    [ulong]$MaximumBytes = 4GB,
    [string]$StatisticsPath
)

# %% isolated package checks

$arguments = @("-q", "--tb=short", "--maxfail=1", "-o", "addopts=", "--confcutdir=test/repository_knowledge_test", "--orm-build", "never") + $TestPaths
& "$PSScriptRoot/run_command.ps1" -Module pytest -Arguments $arguments -Repository $Repository -MinimumFreeBytes $MinimumFreeBytes -MaximumBytes $MaximumBytes -StatisticsPath $StatisticsPath
exit $LASTEXITCODE
