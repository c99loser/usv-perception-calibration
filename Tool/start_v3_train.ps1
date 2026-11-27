$ErrorActionPreference = 'Stop'
$ToolDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DataRoot = 'F:\Challenger program\output\real_lidar_raw_official_stable_annotation_v3_bundle\real_lidar_raw_official_stable_annotation_v3\supervisely_train_raw_project'
$OutputRoot = 'F:\Challenger program\output\real_lidar_raw_official_stable_annotation_v3_bundle\local_annotation_workspace_v3'

$PythonCandidates = @(
    (Join-Path $ToolDir '.venv\Scripts\python.exe'),
    'python',
    'py'
)

foreach ($Candidate in $PythonCandidates) {
    try {
        if ($Candidate -eq 'py') {
            & $Candidate -3 (Join-Path $ToolDir 'run_annotator.py') --data-root $DataRoot --output $OutputRoot
        } else {
            & $Candidate (Join-Path $ToolDir 'run_annotator.py') --data-root $DataRoot --output $OutputRoot
        }
        exit $LASTEXITCODE
    } catch {
        continue
    }
}

Write-Error '没有找到可用的 Python。请先按 README.md 创建 .venv。'

