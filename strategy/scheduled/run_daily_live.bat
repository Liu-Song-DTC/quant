@echo off
REM ============================================================
REM  Daily live chain trigger - weekdays 17:35 Beijing time
REM  Install (admin cmd):
REM    schtasks /Create /TN "QuantDailyLive" /TR "D:\quant\strategy\scheduled\run_daily_live.bat" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 17:35
REM  Remove:
REM    schtasks /Delete /TN "QuantDailyLive" /F
REM  Test now:
REM    D:\quant\strategy\scheduled\run_daily_live.bat
REM  Notes:
REM   - QMT client must stay logged in (Windows 16:00 task updates K-lines)
REM   - No auto-retry by design: failure alerts via WeChat; rerun manually
REM   - Chain log: strategy\logs\scheduler_YYYYMMDD.log
REM ============================================================
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/d/quant && bash strategy/run_daily_live.sh >> strategy/logs/scheduler_$(date +%%Y%%m%%d).log 2>&1"
