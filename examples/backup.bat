@echo off
REM Nightly backup script for the accounting share
setlocal enabledelayedexpansion

set SRC=C:\Data\Accounting
set DEST=D:\Backups\Accounting
set LOG=%DEST%\backup.log

echo Starting backup at %DATE% %TIME% >> %LOG%

if not exist "%DEST%" mkdir "%DEST%"

xcopy "%SRC%" "%DEST%" /E /I /Y /D

REM Purge backups older than 30 days
forfiles /p "%DEST%" /s /m *.bak /d -30 /c "cmd /c del /q @path"

echo Backup finished at %DATE% %TIME% >> %LOG%
endlocal
