[CmdletBinding()]
param(
    [switch]$SkipAnalysis,
    [switch]$SkipPdf
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$demoPython = Join-Path $projectRoot ".venv-demo\Scripts\python.exe"
$coptPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

foreach ($executable in @($demoPython, $coptPython)) {
    if (-not (Test-Path -LiteralPath $executable)) {
        throw "Required Python environment not found: $executable"
    }
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    Write-Host "`n==> $Name" -ForegroundColor Cyan
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Push-Location $projectRoot
try {
    Invoke-Step "Q1 prediction" $demoPython @("scripts/run_q1_demo.py")
    Invoke-Step "Q1 seven-model comparison" $demoPython @("scripts/run_q1_sklearn_challengers.py")
    Invoke-Step "Q1 report" $demoPython @("scripts/build_q1_demo_report.py")
    Invoke-Step "Q2 feasible baseline" $demoPython @("scripts/run_q2_demo.py")
    Invoke-Step "Q2 COPT optimization" $coptPython @("scripts/run_q2_copt.py")
    Invoke-Step "Q3 COPT optimization" $coptPython @("scripts/run_q3_copt.py")
    Invoke-Step "Q4 source parsing" $demoPython @("scripts/collect_q4_openfootball.py")
    Invoke-Step "Q4 structural comparison" $demoPython @("scripts/run_q4_comparison.py")

    if (-not $SkipAnalysis) {
        Invoke-Step "Robustness and convergence analysis" $demoPython @("scripts/run_analysis_campaign.py")
    }

    Invoke-Step "Paper figures" $demoPython @("scripts/build_paper_figures.py")
    Invoke-Step "Python tests" $demoPython @("-m", "pytest", "tests", "-q")
    Invoke-Step "Deliverable validation" $demoPython @("scripts/validate_deliverables.py")

    if (-not $SkipPdf) {
        $env:PYTHONIOENCODING = "utf-8"
        $env:PERL_UNICODE = "SDA"
        Invoke-Step "Main paper PDF" "latexmk" @("-cd", "-g", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", "-outdir=build", "paper/main.tex")
        Invoke-Step "AI usage PDF" "latexmk" @("-cd", "-g", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", "-outdir=build", "paper/ai_usage.tex")
    }
}
finally {
    Pop-Location
}

Write-Host "`nAll requested reproduction steps completed." -ForegroundColor Green

