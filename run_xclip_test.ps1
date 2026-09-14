# ---------------------------------------------------------------------
#  Score an X-CLIP checkpoint on the 14-class task and print the metrics.
#
#      cd D:\UMIL_clean
#      powershell -ExecutionPolicy Bypass -File run_xclip_test.ps1
#
#  Nothing has to be filled in: the config, the checkpoint and the test
#  annotation file are found on disk and echoed before anything runs, so
#  a wrong pick is visible rather than silent. Override any of them:
#
#      ... -Config configs\ucf\x.yaml -Checkpoint exp_v3\best.pth
#
#  If main.py cannot run here (the UMIL folders are missing, or there is
#  no GPU) but a test_scores.pkl exists, the metrics are still printed
#  from that file -- which is all the reported numbers ever came from.
# ---------------------------------------------------------------------

param(
    [string]$Config     = "",
    [string]$Checkpoint = "",
    [string]$Annotations = "",
    [string]$Out        = "exp_v3",
    [int]   $NumClip    = 4
)

function Find-Config {
    $yaml = Get-ChildItem -Recurse -Filter *.yaml -ErrorAction SilentlyContinue

    if (-not $yaml) { return "" }

    # the UCF config is the one declaring fourteen classes
    foreach ($f in $yaml) {
        if (Select-String -Path $f.FullName -Pattern "NUM_CLASSES:\s*14" -Quiet -ErrorAction SilentlyContinue) {
            return $f.FullName
        }
    }

    foreach ($f in $yaml) {
        if ($f.FullName -match "ucf") { return $f.FullName }
    }

    return ""
}

function Find-Newest($filter, $exclude) {
    $hit = Get-ChildItem -Recurse -Filter $filter -ErrorAction SilentlyContinue |
           Where-Object { $exclude -eq "" -or $_.FullName -notmatch $exclude } |
           Sort-Object LastWriteTime -Descending |
           Select-Object -First 1

    if ($hit) { return $hit.FullName }
    return ""
}

Write-Host ""
Write-Host "=== looking for the pieces ===" -ForegroundColor Cyan

if ($Config -eq "")      { $Config      = Find-Config }
if ($Checkpoint -eq "")  { $Checkpoint  = Find-Newest "*.pth" "" }
if ($Annotations -eq "") { $Annotations = Find-Newest "*test*split*.txt" "" }
if ($Annotations -eq "") { $Annotations = Find-Newest "UCF_std_test.txt" "" }

Write-Host ("config      : " + $(if ($Config)      { $Config }      else { "NOT FOUND" }))
Write-Host ("checkpoint  : " + $(if ($Checkpoint)  { $Checkpoint }  else { "NOT FOUND" }))
Write-Host ("annotations : " + $(if ($Annotations) { $Annotations } else { "NOT FOUND" }))
Write-Host ("output      : " + $Out)

$scores = Join-Path $Out "test_scores.pkl"

if ($Annotations -eq "") {
    Write-Host ""
    Write-Host "No test annotation file found. Nothing can be scored against." -ForegroundColor Red
    exit 1
}

# -------------------------------------------------------------------
# step 1 -- run the model, if it can run here at all
# -------------------------------------------------------------------

$canScore = ($Config -ne "") -and ($Checkpoint -ne "") -and (Test-Path "main.py") -and (Test-Path "utils")

if ($canScore) {
    Write-Host ""
    Write-Host "=== scoring the test videos ===" -ForegroundColor Cyan

    python main.py `
        --config $Config `
        --output $Out `
        --only_test `
        --pretrained $Checkpoint `
        --opts TEST.ONLY_TEST True TEST.NUM_CLIP $NumClip

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "main.py failed." -ForegroundColor Yellow

        if (-not (Test-Path $scores)) {
            Write-Host "No existing scores to fall back on either." -ForegroundColor Red
            exit 1
        }

        Write-Host "Falling back to the scores already in $scores" -ForegroundColor Yellow
    }
}
else {
    Write-Host ""
    Write-Host "Skipping main.py -- this folder cannot run it." -ForegroundColor Yellow
    Write-Host "(main.py needs configs/, datasets/, models/ and utils/ next to it.)"

    $found = Find-Newest "test_scores.pkl" ""

    if ($found -eq "") {
        Write-Host ""
        Write-Host "And no test_scores.pkl exists anywhere, so there is nothing to report." -ForegroundColor Red
        Write-Host "Clone the UMIL repository and copy your changed files onto it first."
        exit 1
    }

    $scores = $found
    Write-Host "Using the scores already in $scores"
}

if (-not (Test-Path $scores)) {
    Write-Host ""
    Write-Host "Expected $scores but it is not there." -ForegroundColor Red
    exit 1
}

# -------------------------------------------------------------------
# step 2 -- read the metrics off the scores
# -------------------------------------------------------------------

Write-Host ""
Write-Host "=== 14-class metrics ===" -ForegroundColor Cyan
python evaluate_multiclass.py --scores $scores --annotations $Annotations

Write-Host ""
Write-Host "=== per-class diagnosis ===" -ForegroundColor Cyan
python evaluate_multiclass.py --scores $scores --annotations $Annotations --diagnose

Write-Host ""
Write-Host "=== 95% confidence intervals ===" -ForegroundColor Cyan
python bootstrap_ci.py --scores $scores --annotations $Annotations --iterations 4000

Write-Host ""
Write-Host "Done. Scores: $scores" -ForegroundColor Green
