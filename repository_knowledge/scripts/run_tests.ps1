param(
    [string]$Repository = (Resolve-Path "$PSScriptRoot/../..").Path,
    [string[]]$TestPaths = @("test/repository_knowledge_test"),
    [long]$MinimumFreeBytes = 8GB,
    [ulong]$MaximumBytes = 4GB
)

$availableBytes = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1KB
if ($MaximumBytes -le 0 -or $MaximumBytes -ge $MinimumFreeBytes) {
    throw "The memory cap must be positive and below the minimum available memory."
}
if ($availableBytes -lt $MinimumFreeBytes) {
    throw "Insufficient available memory for the bounded test run."
}
Write-Output ("Free memory: {0:N1} GiB; job limit: {1:N1} GiB." -f ($availableBytes / 1GB), ($MaximumBytes / 1GB))
Add-Type -Path "$PSScriptRoot/memory_job.cs"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$arguments = @("pytest", "-q", "--tb=short", "--maxfail=1", "-o", "addopts=", "--confcutdir=test/repository_knowledge_test", "--orm-build", "never") + $TestPaths
$exitCode = [MemoryJob]::Run(
    "$Repository/repository_knowledge/.venv/Scripts/python.exe",
    "$PSScriptRoot/pytest_worker.py",
    $Repository,
    $arguments,
    $MaximumBytes
)
exit $exitCode
