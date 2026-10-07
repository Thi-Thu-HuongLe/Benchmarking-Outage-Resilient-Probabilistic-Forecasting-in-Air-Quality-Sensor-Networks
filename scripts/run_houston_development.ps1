param(
    [string]$Device = "cuda:0",
    [string]$OutputRoot = "experiment_protocol\reproduction",
    [string]$Protocol = "journal_protocol\third_comparison_protocol.json"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $outputPath = [System.IO.Path]::GetFullPath($OutputRoot, $repoRoot)
    if (Test-Path -LiteralPath $outputPath) {
        throw "Output directory already exists; choose a new -OutputRoot to preserve prior runs: $outputPath"
    }

    $baseRoot = Join-Path $outputPath "base"
    $refinementRoot = Join-Path $outputPath "refinement"
    $analysisRoot = Join-Path $outputPath "development_analysis"

    & python scripts\run_journal_development.py --stage run --device $Device `
        --protocol $Protocol --output $baseRoot
    if ($LASTEXITCODE -ne 0) { throw "Base development run failed with exit code $LASTEXITCODE" }

    & python scripts\run_journal_development.py --stage run --device $Device `
        --protocol $Protocol --learned-models adaptive_graph_mask_tcn fixed_graph_equal_training `
        --without-deterministic --refinement-base-runs (Join-Path $baseRoot "runs") `
        --output $refinementRoot
    if ($LASTEXITCODE -ne 0) { throw "Graph-refinement run failed with exit code $LASTEXITCODE" }

    & python scripts\analyze_third_development.py --protocol $Protocol `
        --base-root $baseRoot --refinement-root $refinementRoot --output $analysisRoot
    if ($LASTEXITCODE -ne 0) { throw "Development analysis failed with exit code $LASTEXITCODE" }

    Write-Host "Houston development workflow completed. Outputs: $outputPath"
}
finally {
    Pop-Location
}
