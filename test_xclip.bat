@echo off
REM ---------------------------------------------------------------
REM  Test an X-CLIP checkpoint on the 14-class task.
REM
REM  Run it from the UMIL working copy (the one with configs/,
REM  datasets/, models/ and utils/ in it), not from the repository
REM  that only holds the changed files:
REM
REM      .\test_xclip.bat
REM
REM  Edit the three values below first.
REM ---------------------------------------------------------------

REM the yaml for UCF in configs/
set CONFIG=configs\ucf\ucf_xclip_b32.yaml

REM the trained checkpoint to score with
set CKPT=exp_v3\checkpoint_best.pth

REM where test_scores.pkl is written
set OUT=exp_v3

REM the annotation file the scores are compared against
set ANNO=labels\UCF_full_test_split.txt

echo.
echo === scoring the test videos ===
python main.py ^
    --config %CONFIG% ^
    --output %OUT% ^
    --only_test ^
    --pretrained %CKPT% ^
    --opts TEST.ONLY_TEST True TEST.NUM_CLIP 4

if errorlevel 1 (
    echo.
    echo main.py failed. Check CONFIG and CKPT above.
    exit /b 1
)

echo.
echo === 14-class metrics ===
python evaluate_multiclass.py --scores %OUT%\test_scores.pkl --annotations %ANNO%

echo.
echo === per-class diagnosis ===
python evaluate_multiclass.py --scores %OUT%\test_scores.pkl --annotations %ANNO% --diagnose

echo.
echo === 95%% confidence intervals ===
python bootstrap_ci.py --scores %OUT%\test_scores.pkl --annotations %ANNO% --iterations 4000

echo.
echo Done. Scores are in %OUT%\test_scores.pkl
