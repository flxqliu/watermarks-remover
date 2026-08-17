\# Windows: Auto-start the service at login



Run the service silently in the background on every login, without a terminal window or manual start.



\## 1. Clone the repo somewhere permanent



\\`\\`\\`powershell

git clone https://github.com/guillaumemeyer/watermarks-remover.git C:\\watermarks-remover

\\`\\`\\`



\## 2. Create a silent launcher script



Save as `C:\\watermarks-remover\\start-service.vbs`:



\\`\\`\\`vbscript

Set WshShell = CreateObject("WScript.Shell")

WshShell.Run "python service\\scripts\\server.py --host 127.0.0.1 --port 8765", 0, False

\\`\\`\\`



\## 3. Register a scheduled task (run PowerShell as Administrator)



\\`\\`\\`powershell

$action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument '"C:\\watermarks-remover\\start-service.vbs"' -WorkingDirectory "C:\\watermarks-remover"

$trigger = New-ScheduledTaskTrigger -AtLogOn

$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName "WatermarksRemoverService" -Action $action -Trigger $trigger -Settings $settings -Description "Auto-starts the watermarks-remover HTTP service at login"

\\`\\`\\`



\## 4. Start it immediately (no reboot needed)



\\`\\`\\`powershell

Start-ScheduledTask -TaskName "WatermarksRemoverService"

\\`\\`\\`



\## 5. Verify



\\`\\`\\`powershell

Invoke-RestMethod http://127.0.0.1:8765/health

\\`\\`\\`



Should return `{"ok": true, "version": "..."}`.



\## Notes



\- Requires Python 3.10+ on PATH.

\- The scheduled task runs at every login going forward — no manual start needed.

\- To stop auto-starting: `Unregister-ScheduledTask -TaskName "WatermarksRemoverService"`

