@echo off
setlocal

set "WORKSPACE=/mnt/f/Challenger program"
set "SCRIPT=tools/camera_fisheye/verify_ros2_corrected_mcaps.sh"
set "WSL=%SystemRoot%\System32\wsl.exe"

echo [ros2-verify] Trying Ubuntu-24.04...
"%WSL%" -d Ubuntu-24.04 --exec bash -lc "cd '%WORKSPACE%' && bash '%SCRIPT%'"
if %ERRORLEVEL% EQU 0 goto :done

echo [ros2-verify] Ubuntu-24.04 failed with exit code %ERRORLEVEL%.
echo [ros2-verify] Trying Ubuntu...
"%WSL%" -d Ubuntu --exec bash -lc "cd '%WORKSPACE%' && bash '%SCRIPT%'"
if %ERRORLEVEL% EQU 0 goto :done

echo [ros2-verify] Both Ubuntu-24.04 and Ubuntu failed.
exit /b %ERRORLEVEL%

:done
echo [ros2-verify] Done.
echo [ros2-verify] Report: F:\Challenger program\output\camera_fisheye\ros2_verify\ros2_corrected_mcap_verify.txt
