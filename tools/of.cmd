@echo off
setlocal
rem Run OpenFOAM 12 in the current case directory.
set "HIMCM_CASE_DIR=%CD%"
if defined BLUECFD_ROOT (
    if not exist "%BLUECFD_ROOT%\setvars_OF12.bat" (
        echo ERROR: BLUECFD_ROOT must contain setvars_OF12.bat. 1>&2
        exit /b 2
    )
    call "%BLUECFD_ROOT%\setvars_OF12.bat"
    if errorlevel 1 exit /b 2
)
cd /d "%HIMCM_CASE_DIR%"
where foamRun >nul 2>nul
if errorlevel 1 (
    echo ERROR: Set BLUECFD_ROOT to a blueCFD-Core installation containing OpenFOAM 12. 1>&2
    exit /b 2
)
if "%~1"=="" (
    echo Usage: of.cmd command [arguments] 1>&2
    exit /b 2
)
%*
exit /b %errorlevel%
