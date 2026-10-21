# %% kernel connection

param(
    [Parameter(Mandatory = $true)][string]$ConnectionFile,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$KernelArguments = @()
)

# %% bounded kernel

& "$PSScriptRoot/run_command.ps1" -Module ipykernel_launcher -Arguments (@('-f', $ConnectionFile) + $KernelArguments)
exit $LASTEXITCODE
