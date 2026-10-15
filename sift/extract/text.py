"""Extractor for plain text, markdown, logs, source code and shell/batch scripts."""
from __future__ import annotations

import re
from pathlib import Path

from ..detect import Detection
from .base import Extracted, clip, decode_text

MAX_TEXT_CHARS = 200_000


_LINE_END = re.compile(r"\r\n|\r|\n")


def extract(path: Path, det: Detection, max_bytes: int) -> Extracted:
    with path.open("rb") as fh:
        data = fh.read(max_bytes)
    cut_short = det.size > len(data)
    text, encoding = decode_text(data, cut_short)
    body, truncated = clip(text, MAX_TEXT_CHARS)

    # Not splitlines(): it also breaks on \x0b \x0c \x1c-\x1e \x85 U+2028 and
    # U+2029, so a log with one form feed reported four lines where wc -l says
    # three. A line ends at a line ending.
    lines = _LINE_END.split(text)
    if lines and lines[-1] == "":
        lines.pop()                     # a trailing newline ends the last line
    meta = {
        "encoding": encoding,
        "lines": len(lines),
        "characters": len(text),
        "words": len(text.split()),
        "blank_lines": sum(1 for ln in lines if not ln.strip()),
    }
    if det.language:
        meta["language"] = det.language

    if det.kind in ("batch", "shell", "powershell") or det.language in (
            "batch", "shell", "powershell"):
        meta.update(_script_signals(text, det.kind))
    elif det.kind == "source-code":
        meta.update(_code_signals(text))
    elif det.kind == "markdown":
        meta.update(_markdown_signals(text))
    elif det.kind == "log":
        meta.update(_log_signals(lines))

    ex = Extracted(kind=det.kind, mime=det.mime, text=body, metadata=meta,
                   truncated=truncated)
    if det.kind == "log":
        ex.digest = _log_digest(lines, meta, partial=det.size > max_bytes)
    if det.size > max_bytes:
        ex.note(f"file larger than max_bytes ({det.size} bytes); read first {max_bytes}")
    return ex


#: Ordinary lines kept for context around the flagged ones.
_DIGEST_SAMPLE = 40


def _log_digest(lines: list[str], meta: dict,
                partial: bool = False) -> str:
    """The parts of a log worth a model's attention.

    A log is mostly repetition. Sending all of it costs one model call per
    20,000 characters and buries the four lines that matter; sending the
    counts, every flagged line and a sample of the routine ones costs one call
    and shows the model the signal instead of the noise.
    """
    flagged = {ln.strip() for ln in lines if _LEVEL.search(ln)}
    ordinary = [ln.strip() for ln in lines if ln.strip() and ln.strip() not in flagged]
    step = max(1, len(ordinary) // _DIGEST_SAMPLE)
    sample = ordinary[::step][:_DIGEST_SAMPLE]
    counts = meta.get("level_counts", {})

    # Distinct lines with their occurrence counts, not one entry per
    # occurrence. A log of 200 identical "ERROR connection refused" lines
    # announced "Every flagged line (1)" and then printed all 200 — a header
    # contradicting its own body, and the repetition crowding out everything
    # the digest exists to make room for.
    occurrences: dict[str, int] = {}
    for ln in lines:
        stripped = ln.strip()
        if stripped in flagged:
            occurrences[stripped] = occurrences.get(stripped, 0) + 1
    total_flagged = sum(occurrences.values())

    # Every count below is over what was read. On a file past max_bytes that
    # is a prefix, and saying "across the whole file" about the first 200 KB of
    # a 1 MB log stated a blank severity table and "0 flagged lines" while
    # thirty ERROR lines sat past the read window.
    scope = "the part read" if partial else "the whole file"
    parts = [
        f"# A log file, of which {len(lines)} lines were read"
        if partial else f"# A log file of {len(lines)} lines",
        "You are shown a representative extract of it, not every line: all "
        "flagged lines are here, ordinary ones are sampled evenly. Describe "
        "the LOG FILE, not the extract — the extract is how you are reading "
        "it, not what it is." + (" The file is larger than the amount read, "
        "so the counts below cover its beginning only." if partial else ""),
        "",
        f"## Severity counts across {scope}",
        ", ".join(f"{k}: {v}" for k, v in counts.items()) or "none",
        "",
        f"## Every flagged line ({total_flagged} occurrences, "
        f"{len(occurrences)} distinct)",
    ]
    ordered = [f"{line}    [× {n}]" if n > 1 else line
               for line, n in occurrences.items()]
    parts.extend(ordered[:400])
    if len(ordered) > 400:
        parts.append(f"… and {len(ordered) - 400} more distinct flagged lines")
    parts += ["", f"## {len(sample)} ordinary lines, evenly spaced", *sample]
    if lines:
        parts += ["", "## First and last line", lines[0].strip(), lines[-1].strip()]
    return "\n".join(parts)


# ---- signal extractors ------------------------------------------------
def _cmd(name: str) -> str:
    """A command name, with or without the .exe the shell supplies anyway.

    `reg add` was flagged and `reg.exe add` was not — the plainest evasion
    there is, and it needs no cleverness at all: half the scripts on the
    internet write the suffix out of habit.
    """
    return rf"\b{name}(?:\.exe)?\b"


#: What a script does that is worth naming. Measured against 31 hallmark
#: destructive lines: the first version of this list caught 11. The misses
#: were not exotic — deleting shadow copies is the opening move of every
#: ransomware script written since 2016, and it was missed because
#: `\b(del|...)\b` cannot match inside the word "delete".
_DANGER = [
    (rf"{_cmd('del')}.*(/s|/q|\*)|{_cmd('rd')}.*(/s|/q)"
     rf"|{_cmd('rmdir')}.*(/s|/q)|{_cmd('erase')}.*(/s|/q|\*)",
     "recursive/forced delete"),
    # The r matters, not the f: "rm -f build/.lock" removes one file and was
    # reported as a recursive force remove, as was `trap 'rm -f "$TMP"' EXIT`
    # in every careful script ever written. A wildcard or an absolute path
    # under a system directory is the other shape worth naming.
    # Long-form GNU flags are the same command spelled out.
    (r"\brm\s+(?:-[a-z]*r[a-z]*|--recursive)\s"
     r"|\brm\s+-[a-z]*f[a-z]*\s+(?:/(?:etc|var|usr|opt|home|srv|boot)\b|[^\n]*\*)",
     "recursive force remove"),
    (rf"{_cmd('format')}\s+[a-z]:", "disk format"),
    (rf"{_cmd('reg')}\s+(add|delete|import)\b", "registry modification"),
    (rf"{_cmd('shutdown')}|{_cmd('Restart-Computer')}", "system shutdown"),
    (rf"{_cmd('diskpart')}", "disk partitioning"),
    (rf"{_cmd('curl')}|{_cmd('wget')}|\bInvoke-WebRequest\b|\biwr\b"
     rf"|{_cmd('bitsadmin')}|\bStart-BitsTransfer\b"
     rf"|{_cmd('certutil')}[^\n]*-urlcache"
     r"|\bDownloadString\b|\bDownloadFile\b|\bWebClient\b"
     r"|\birm\b[^\n]*-OutFile|\bcurl\b[^\n]*\s-[a-zA-Z]*[oO]\s"
     r"|\bwget\b[^\n]*\s-[a-zA-Z]*O\s",
     "network download"),
    (rf"{_cmd('net')}\s+user\b|{_cmd('net')}\s+localgroup\b"
     rf"|\bNew-LocalUser\b|\bAdd-LocalGroupMember\b",
     "account modification"),
    # Stopping your own service before replacing it is what a deploy script
    # does. The force flag is what separates that from killing something
    # that does not want to stop.
    # A deploy script stops its own service; killing a security product is
    # a different act, and naming those makes the force flag unnecessary.
    (rf"{_cmd('taskkill')}[^\n]*\s/f\b|\bStop-Process\b[^\n]*-Force"
     r"|\bStop-Service\b[^\n]*-Force"
     r"|\bkill(?:all)?\s+-(?:9|KILL)\b|\bpkill\s+-(?:9|KILL)\b"
     r"|\b(?:taskkill|Stop-Process)\b[^\n]*(?:MsMpEng|SentinelAgent|CSFalcon"
     r"|CrowdStrike|Sophos|McAfee|CylanceSvc)", "process termination"),
    # iex is PowerShell's Invoke-Expression *applied to something*; bare
    # "iex" also starts Elixir's REPL, which is an ordinary line in a CI
    # script. And "-e <base64>" only means an encoded command when powershell
    # is what is being run — otherwise it is grep's or sed's expression flag,
    # and a secret-scanning line was reported as obfuscated execution.
    # PowerShell accepts every unambiguous prefix of -EncodedCommand, and
    # the shipping binary of PowerShell 7 is pwsh, not powershell. Two
    # spellings of the flag and one of the program missed both.
    (r"\bInvoke-Expression\b|\biex\s*[(\"'$@]|\|\s*iex\b"
     r"|-Encoded(?:Command)?\b"
     r"|\b(?:powershell|pwsh)(?:\.exe)?[^\n]*\s-e[a-z]*\s+"
     r"[A-Za-z0-9+/=]{16,}"
     r"|\[scriptblock\]::Create|FromBase64String"
     rf"|{_cmd('certutil')}[^\n]*-decode", "dynamic/obfuscated execution"),
    # Everything below was missed entirely. Each is a step a real script takes
    # and an ordinary one does not.
    # Every alternative needs the verb. "(Get-WmiObject Win32_Shadowcopy)
    # .Count" counts them, and labelling a count a deletion is worse than
    # saying nothing: the label is what reaches the reader and the model.
    (rf"{_cmd('vssadmin')}[^\n]*\b(?:delete|resize)\b"
     rf"|{_cmd('wmic')}[^\n]*shadowcopy[^\n]*delete"
     rf"|{_cmd('wbadmin')}[^\n]*\bdelete\b"
     r"|\b(?:Get-WmiObject|Get-CimInstance)\b[^\n]*Win32_Shadowcopy\b[^\n]*"
     r"(?:Remove-CimInstance|Remove-WmiObject|\.Delete)",
     "deletes backups or shadow copies"),
    (rf"{_cmd('bcdedit')}[^\n]*(recoveryenabled|bootstatuspolicy|safeboot)",
     "disables Windows recovery"),
    (rf"{_cmd('schtasks')}\s*/create|\bRegister-ScheduledTask\b"
     # `reg query …\Run` reads the key. The sibling alternatives all name a
     # verb; this one did not, so a monitoring script that lists autostart
     # entries was reported as installing itself.
     rf"|{_cmd('reg')}\s+(?:add|delete|import)[^\n]*\\(?:Run|RunOnce)\b"
     r"|\\Start\s?Menu\\Programs\\Startup"
     # `systemctl enable myapp` is what an installer does with its own
     # service; cron entries and Run keys are what a payload does with
     # itself.
     r"|(?:crontab\s+-|/etc/cron\.[a-z]+/|/var/spool/cron)"
     r"|\b(?:New|Set)-ItemProperty\b[^\n]*(?:Run|RunOnce)\b",
     "installs itself to run again"),
    # Adding a firewall rule is what an installer does. Turning the firewall
    # off is not, and only the second was meant.
    (rf"{_cmd('netsh')}[^\n]*(?:advfirewall\s+reset"
     rf"|firewall[^\n]*\b(?:state\s+off|opmode\s+disable|set\s+opmode"
     rf"|disable|allowinbound)\b)"
     r"|\bSet-NetFirewallProfile\b[^\n]*-Enabled\s+False"
     rf"|{_cmd('sc')}\s+(?:stop|config|delete)\s+[^\n]*"
     r"(?:windefend|wuauserv|wscsvc|sense|sophos|mcafee)"
     rf"|{_cmd('net')}\s+stop\s+[^\n]*(?:windefend|wuauserv|wscsvc)"
     r"|\bSet-MpPreference\b[^\n]*-Disable|\bAdd-MpPreference\b"
     # `service auditd stop` puts the name before the verb; systemd puts it
     # after. Both orders are the same act.
     r"|\b(?:systemctl|service)\s+(?:(?:stop|disable|mask)\s+)?"
     r"(?:auditd|firewalld|apparmor|ufw|falco|osquery|rsyslog|syslog|wazuh"
     r"|crowdstrike|sentinel|clamav)[\w.-]*(?:\s+(?:stop|disable|mask))?\b"
     r"|\biptables\s+(?:-F|--flush)\b|\bsetenforce\s+0\b"
     r"|\bufw\s+disable\b"
     rf"|{_cmd('net')}\s+stop\s+[\"']?[^\n]*(?:defend|sophos|mcafee|sentinel"
     r"|crowdstrike|antivirus)",
     "turns off a security service"),
    (rf"{_cmd('wevtutil')}\s+cl\b|\bClear-EventLog\b"
     rf"|{_cmd('fsutil')}[^\n]*deletejournal|{_cmd('cipher')}\s+/w",
     "erases traces"),
    # An installer grants its own service account rights and hides its
    # uninstall data; that is not what this is for. Taking ownership of
    # someone else's tree, granting Everyone, and the +h +s pair that hides a
    # file from the shell entirely are.
    # attrib +h on an executable hides a payload; on uninstall.dat it is an
    # installer. Requiring two flags took +h, +s and +h +r with it — the
    # narrowing was aimed at the object, not at the flag count. And icacls
    # writes /grant:r as often as /grant, which the pattern could not match.
    (rf"{_cmd('takeown')}|{_cmd('icacls')}[^\n]*(?:/setowner"
     r"|/grant:?r?\s*[^\n]*(?:everyone|users|authenticated|:\(?F\)?)"
     r"|/inheritance:r)"
     rf"|{_cmd('cacls')}"
     rf"|{_cmd('attrib')}[^\n]*\+\s*[hs][^\n]*"
     r"(?:\+\s*[hs]|\.(?:exe|dll|scr|bat|cmd|vbs|ps1|js|jar)\b)"
     r"|\bchmod\s+(?:-R\s+(?:000|777)|(?:000|777)\s+-R)\b"
     r"|\bchown\s+-R\s+root\b"
     r"|\bchattr\s+\+i\b",
     "changes ownership or hides files"),
    # Everything above grew out of Windows batch. A shell script and a
    # PowerShell script do the same things with different words, and the
    # commit that widened this list measured only the batch side: a complete
    # Linux ransomware script — encrypt in place, shred the originals, cron
    # persistence, stop auditd, truncate the logs, wipe a disk — came back
    # with "None of the operations sift watches for appear in it".
    # Argument order is the writer's choice, and NVMe has been the ordinary
    # disk name for a decade.
    (r"\bdd\b[^\n]*\bof=/dev/(?:[sh]d[a-z]|nvme\d|vd[a-z]|disk\d)"
     r"|\bmkfs(?:\.[a-z0-9]+)?\s|\bwipefs\b|\bblkdiscard\b"
     r"|>\s*/dev/(?:[sh]d[a-z]|nvme\d+(?:n\d+)?|vd[a-z])", "disk format"),
    (r"\bshred\b|\bsrm\b|\bwipe\s+-[a-z]*r"
     r"|\bhistory\s+-c\b|\bunset\s+HISTFILE\b"
     r"|\btruncate\s+-s\s*0\s+/var/log"
     r"|>\s*/var/log/\w+|\brm\s+-[a-z]*f?[a-z]*\s+~?/\.bash_history",
     "erases traces"),
    (r"\bopenssl\s+(?:enc\b|aes-\d+)"
     r"|\bgpg\b[^\n]*--(?:symmetric|encrypt)\b"
     r"|\b7z\s+a\b[^\n]*-p", "encrypts files in place"),
    (r"\buseradd\b|\busermod\b[^\n]*-aG\s+(?:sudo|wheel|admin)"
     r"|\bauthorized_keys\b|\bpasswd\s+\w+\s*<|\bvisudo\b"
     r"|/etc/sudoers(?:\.d)?\b"
     r"|\bNew-LocalUser\b|\bAdd-LocalGroupMember\b",
     "account modification"),
    (r"\bRemove-Item\b[^\n]*-Recurse[^\n]*-Force"
     r"|\bRemove-Item\b[^\n]*-Force[^\n]*-Recurse"
     r"|\|\s*Remove-Item\b[^\n]*-(?:Force|Recurse)"
     r"|\bri\s+-r[a-z]*\s+-fo", "recursive/forced delete"),
    (r"\bSet-ExecutionPolicy\b[^\n]*(?:Bypass|Unrestricted)"
     r"|\bStart-Process\b[^\n]*-Verb\s+RunAs|\bsudo\s+-S\b",
     "raises its own privileges"),
    (r"\bInvoke-CimMethod\b[^\n]*Win32_Process[^\n]*Create"
     r"|\bbash\s+-i\b[^\n]*/dev/tcp/|\b(?:nc|ncat)\b[^\n]*\s-e\s"
     rf"|{_cmd('wscript')}|{_cmd('cscript')}|{_cmd('msiexec')}[^\n]*https?://"
     rf"|{_cmd('esentutl')}[^\n]*/y|{_cmd('forfiles')}[^\n]*/c\s+\"?cmd"
     rf"|{_cmd('regedit')}\s+/s|{_cmd('sc')}(?:\.exe)?\s+create",
     "runs code through another program"),
    (rf"{_cmd('mshta')}|{_cmd('rundll32')}|{_cmd('regsvr32')}"
     rf"|{_cmd('wmic')}[^\n]*process[^\n]*call\s+create",
     "runs code through another program"),
    # robocopy to a UNC path is the nightly backup of every Windows shop,
    # and `net use Z: \\server\share` is how it gets mounted. Neither is
    # worth a warning. Piping an archive into a socket is.
    (r"\btar\s+[a-z]*c[a-z]*\b[^\n]*\|[^\n]*\b(?:nc|ncat|netcat|curl|wget)\b"
     r"|\bscp\s+-r\s+[^\n]*@|\b(?:nc|ncat)\s+[^\n]*\s\d{2,5}\s*<"
     r"|\bcurl\b[^\n]*(?:--upload-file|-T\s)"
     r"|\bInvoke-RestMethod\b[^\n]*-Method\s+Post[^\n]*-InFile",
     "moves data over the network"),
]

#: In batch, REM and :: introduce a comment; in the others, #. Everything a
#: shell would not run has to come out before the scan, because a comment
#: naming a command is not that command: "REM Do NOT use curl here, the proxy
#: blocks it" was reported as a network download, and a backup script was
#: warned about on the strength of a sentence explaining that it downloads
#: nothing.
_COMMENT = {
    "batch": re.compile(r"^\s*(?:rem\b|::)", re.IGNORECASE),
    "shell": re.compile(r"^\s*#"),
    "powershell": re.compile(r"^\s*#"),
}
#: What opens a region a shell reads as text rather than as code, and what
#: closes it. A line scanner rather than regexes over the whole file: the
#: regex version could be opened from inside a string — `$open = "<#"` ate
#: four lines of six — and from inside a comment, where a `#` line mentioning
#: `<<EOF` swallowed the script's own later heredoc as its terminator. And it
#: had a rule for `goto :eof` that was simply wrong: that is how batch returns
#: from a `call :label` subroutine, so it sits above every subroutine in every
#: structured batch file, and deleting from there to the end silenced a
#: ransomware script down to "None of the operations sift watches for appear
#: in it" — this file's own headline failure, reintroduced by the feature
#: added to fix it.
_BLOCK_OPENERS = {
    "powershell": [
        (re.compile(r"^\s*<#"), "#>"),
        (re.compile(r"@'\s*$"), "'@"),
        (re.compile(r'@"\s*$'), '"@'),
    ],
    "shell": [(re.compile(r"<<-?\s*['\"]?(\w+)['\"]?\s*$"), None)],
}


def _opens_text_block(line: str, kind: str):
    """The terminator of a text region this line opens, or None."""
    for pattern, closer in _BLOCK_OPENERS.get(kind, ()):
        match = pattern.search(line)
        if match:
            return closer if closer is not None else match.group(1)
    return None


_ECHO_HEAD = re.compile(
    r"^\s*(?:@?echo|Write-(?:Output|Host|Verbose|Information|Warning)|printf)"
    r"\b", re.IGNORECASE)


def _strip_echo(line: str) -> str:
    """Remove the part of the line that only prints.

    Quote-aware: `;` ends a statement, but not inside a string. Written as a
    regex that stopped at `;` unconditionally, an installer's own
    documentation — echo "Options: --purge; runs rm -rf /var/lib/myapp on
    every host" — became a recursive delete.
    """
    match = _ECHO_HEAD.match(line)
    if not match:
        return line
    rest = line[match.end():]
    quote = None
    for index, char in enumerate(rest):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
        elif char in "|>&;":
            return rest[index:]          # from the separator on, it runs
    return ""


def _executable_text(text: str, kind: str) -> str:
    """The part of the script a shell would act on.

    Comments are asked about before block openers, so a comment cannot open a
    here-string; and a block that never closes is restored rather than
    swallowed, because "the rest of the file is a comment" is a claim a
    single stray `<#` should not be able to make.
    """
    comment = _COMMENT.get(kind)
    out: list = []
    open_at, closer = None, None
    for number, line in enumerate(text.splitlines()):
        if closer is not None:
            out.append("")
            if line.strip().startswith(closer) or line.strip() == closer:
                closer = None
            continue
        if comment is not None and comment.match(line):
            out.append("")
            continue
        found = _opens_text_block(line, kind)
        if found is not None:
            open_at, closer = number, found
            out.append("")
            continue
        out.append(_strip_echo(line))
    if closer is not None and open_at is not None:
        # Never terminated: put back everything it ate.
        original = text.splitlines()
        out[open_at:] = original[open_at:]
    return "\n".join(out)


def _script_signals(text: str, kind: str) -> dict:
    runnable = _executable_text(text, kind)
    warnings = [label for pattern, label in _DANGER
                if re.search(pattern, runnable, re.IGNORECASE)]
    return {
        "script_kind": kind,
        "commands_sampled": _first_commands(text),
        "risk_signals": sorted(set(warnings)),
    }


#: Lines that are setup, not action. Listing them as "key points" filled the
#: summary of a backup script with "@echo off" and "setlocal" while the xcopy
#: and the delete — the things the script actually does — never appeared.
_BOILERPLATE = re.compile(
    r"^\s*(@?echo\s+(off|on)\s*$|setlocal|endlocal|set\s+\"?[A-Za-z_]\w*=|"
    r"^#!|export\s+\w+=|shopt\s|Set-StrictMode|\[CmdletBinding|param\s*\(|"
    r"chcp\s|pushd|popd|cls\s*$|clear\s*$"
    # A bare assignment and a shell option are preparation too. Without them
    # a backup script's key points were SRC=/srv/data and DEST=/mnt/backup
    # while the tar and the retention sweep — what it is for — came last or
    # not at all.
    r"|set\s+[-+][a-zA-Z]+\s*$|[A-Za-z_]\w*=[^=]|\$[A-Za-z_]\w*\s*="
    r"|\bLC_ALL=|umask\s|IFS=)", re.IGNORECASE)


#: Lines that are the shape of a script rather than a thing it does: a label,
#: a branch, a closing brace, a return. Listed as "key points" they filled a
#: deploy script's summary with ")" and "exit /b 2" while the xcopy and the
#: service restart — the things it does — never appeared.
_PUNCTUATION_LINE = re.compile(
    r"^[)\]}\s]*$|^:[\w.-]+$|^(?:fi|done|esac|end|endlocal|else|do|then)\b"
    r"|^\}\s*(?:else|catch|finally)?\s*\{?$"
    r"|^(?:exit|return)\b[^|&;]*$"
    r"|^(?:if|elif|else\s+if)\b.{0,12}$"
    r"|^\s*(?:goto|call)\s*:\w+\s*$", re.IGNORECASE)


def _first_commands(text: str, limit: int = 8) -> list[str]:
    """The lines that do something, preferred over the lines that prepare."""
    actions, setup, shape = [], [], []
    for ln in _LINE_END.split(text):
        s = ln.strip()
        if not s or s.startswith(("#", "::", "rem ", "REM ")):
            continue
        if _PUNCTUATION_LINE.match(s):
            shape.append(s[:120])
        elif _BOILERPLATE.match(s):
            setup.append(s[:120])
        else:
            actions.append(s[:120])
        if len(actions) >= limit:
            break
    return (actions or setup or shape)[:limit]


def _code_signals(text: str) -> dict:
    return {
        "functions": len(re.findall(r"^\s*(def |function |func |fn |sub )",
                                    text, re.MULTILINE)),
        "classes": len(re.findall(r"^\s*(class |interface |struct )",
                                  text, re.MULTILINE)),
        "imports": len(re.findall(r"^\s*(import |from |#include|require\(|use )",
                                  text, re.MULTILINE)),
        "todos": len(re.findall(r"\b(TODO|FIXME|XXX|HACK)\b", text)),
    }


def _markdown_signals(text: str) -> dict:
    headings = re.findall(r"^(#{1,6})\s+(.*)$", text, re.MULTILINE)
    return {
        "headings": [h[1].strip()[:80] for h in headings[:12]],
        "heading_count": len(headings),
        "code_blocks": text.count("```") // 2,
        "links": len(re.findall(r"\[[^\]]+\]\([^)]+\)", text)),
    }


#: A level word is a token on its own, not a syllable of a path. "\b" treats
#: "-" and "/" as boundaries, so 20 access-log lines requesting /error-page
#: were reported as 20 errors; and the list knew nothing of SEVERE, ALERT,
#: EMERG or syslog's <crit>, so five "SEVERE: batch aborted" lines came back
#: as "No errors or warnings detected".
_LEVEL_WORDS = (r"ERROR|ERR|WARN(?:ING)?|CRITICAL|CRIT|FATAL|SEVERE|ALERT"
                r"|EMERG(?:ENCY)?|PANIC|EXCEPTION|TRACEBACK")
_LEVEL = re.compile(rf"(?<![\w/.=?&-])(?:<)?({_LEVEL_WORDS})(?:>)?"
                    rf"(?![\w/.=?&-])", re.IGNORECASE)


#: Worst first. A log is read for its worst moment, and taking the first six
#: matches meant a CRITICAL in the middle of a 4,000-line file never appeared
#: while six routine warnings from the first minute did.
_SEVERITY = {"PANIC": 0, "EMERG": 0, "EMERGENCY": 0, "FATAL": 0,
             "ALERT": 1, "CRITICAL": 1, "CRIT": 1, "SEVERE": 1,
             "EXCEPTION": 2, "TRACEBACK": 2,
             "ERROR": 3, "ERR": 3, "WARNING": 4, "WARN": 4}


#: The routine levels. They are not "flagged", but leaving them out of a key
#: called level_counts made it read as a full histogram while reporting only
#: the bad news — a log of 4,000 INFO lines and 3 errors looked, in the
#: metadata, like a log of 3 lines.
_ROUTINE = re.compile(r"(?<![\w/.=?&-])(?:<)?(INFO|DEBUG|NOTICE|TRACE|VERBOSE)"
                      r"(?:>)?(?![\w/.=?&-])", re.IGNORECASE)


#: How far into a line the level may sit. A timestamp, a host, a process id
#: and a component name fit comfortably; a message body does not.
_LEVEL_HEAD = 72


def _log_signals(lines: list[str]) -> dict:
    levels: dict[str, int] = {}
    hits: list[tuple[int, int, str]] = []
    for number, ln in enumerate(lines):
        # The level sits in the head of the line, after the timestamp — and
        # whichever list it is in, the *first* one found is the line's level.
        #
        # Searching the whole line, worst-list first, made the message body
        # decide: "INFO Retrying after ERROR 500 from upstream" was counted
        # as an error, so a log with no error line at all was summarized as
        # "43 lines, 41 info lines and 2 errors" — a number a reader would
        # act on, about a file that does not contain it.
        head = ln[:_LEVEL_HEAD]
        bad = _LEVEL.search(head)
        routine = _ROUTINE.search(head)
        if bad and (not routine or bad.start() <= routine.start()):
            key = bad.group(1).upper()
            levels[key] = levels.get(key, 0) + 1
            hits.append((_SEVERITY.get(key, 5), number, ln.strip()[:160]))
            continue
        if routine:
            key = routine.group(1).upper()
            levels[key] = levels.get(key, 0) + 1
    hits.sort()                          # severity, then order of appearance
    samples = [text for _, _, text in hits[:6]]
    meta = {"level_counts": levels, "error_samples": samples}
    if len(hits) > len(samples):
        meta["error_samples_note"] = (
            f"the {len(samples)} most severe of {len(hits)} flagged lines")
    return meta
