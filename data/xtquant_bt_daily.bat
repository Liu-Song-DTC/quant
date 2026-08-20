@echo off
rem 每日 16:00 定时任务: 增量下载A股日线并同步 backtrader 数据
rem 由 Windows 计划任务 "xtquant_bt_daily" 调用, 需 QMT 客户端已登录
cd /d D:\quant
echo ======================================== >> strategy\logs\xtquant_bt.log
echo %date% %time% xtquant downloader --bt start >> strategy\logs\xtquant_bt.log
"C:\Users\admin\AppData\Local\Programs\Python\Python310\python.exe" data\xtquant_downloader.py --bt >> strategy\logs\xtquant_bt.log 2>&1
echo %date% %time% xtquant downloader --bt done (exit=%errorlevel%) >> strategy\logs\xtquant_bt.log
