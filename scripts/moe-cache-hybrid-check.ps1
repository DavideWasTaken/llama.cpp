# Checks the hybrid MoE cache on a CUDA machine, e.g. before merging a sync PR from upstream:
#   1. test-mul-mat-id-skip (CPU kernel)
#   2. perplexity with ubatch 16 (hybrid path), hybrid vs baseline within MaxPplDiff
#   3. determinism: two hybrid runs give the same text
#   4. speed: hybrid vs baseline generation (a warning only, it depends on the hardware)
# Exit code 0 if 1-3 pass.
#
# usage: .\scripts\moe-cache-hybrid-check.ps1 -Model <model.gguf> [-Bin build\bin] [-NCpuMoeBase 33] [-NCpuMoe 40] [-CacheMiB 3200]

param(
    [Parameter(Mandatory)][string]$Model,
    [string]$Bin         = (Join-Path $PSScriptRoot "..\build\bin"),
    [int]   $NCpuMoeBase = 33,
    [int]   $NCpuMoe     = 40,
    [int]   $CacheMiB    = 3200,
    [int]   $Threads     = 8,
    [double]$MaxPplDiff  = 0.01
)

$ErrorActionPreference = "Stop"
$work = Join-Path ([System.IO.Path]::GetTempPath()) "moe-cache-hybrid-check"
New-Item -ItemType Directory -Force $work | Out-Null

$common = @("-m", $Model, "-ngl", "99", "-fa", "on", "-t", "$Threads", "-lm", "none", "-ctk", "q8_0", "-ctv", "q8_0")
$base   = @("--n-cpu-moe", "$NCpuMoeBase")
$hybrid = @("--n-cpu-moe", "$NCpuMoe", "--moe-cache-mib", "$CacheMiB", "--moe-cache-hybrid")

$failed = 0
function Report([string]$name, [bool]$ok, [string]$detail) {
    $script:failed += [int](-not $ok)
    Write-Host ("{0,-12} {1,-5} {2}" -f $name, ($(if ($ok) { "PASS" } else { "FAIL" })), $detail)
}

# runs a llama.cpp tool, returns its stderr log as lines and writes stdout to $out
function Invoke-Tool([string]$tool, [string[]]$toolArgs, [string]$name) {
    $quoted = $toolArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    $log = Join-Path $work "$name.log"
    $out = Join-Path $work "$name.out"
    $p = Start-Process -FilePath (Join-Path $Bin "$tool.exe") -ArgumentList $quoted -Wait -PassThru -NoNewWindow `
            -RedirectStandardError $log -RedirectStandardOutput $out
    if ($p.ExitCode -ne 0) {
        Get-Content $log -Tail 5 | Write-Host
        throw "$tool failed ($name), log: $log"
    }
    return Get-Content $log
}

function Get-Ppl([string[]]$lines) {
    $m = $lines | Select-String "Final estimate: PPL = ([\d.]+)" | Select-Object -Last 1
    return [double]$m.Matches[0].Groups[1].Value
}

function Get-Tps([string[]]$lines) {
    $m = $lines | Select-String "eval time" | Where-Object { $_.Line -notmatch "prompt eval" } | Select-Object -Last 1
    return [double]([regex]::Match($m.Line, "([\d.]+) tokens per second").Groups[1].Value)
}

# 1. CPU kernel test
$test = Join-Path $Bin "test-mul-mat-id-skip.exe"
if (Test-Path $test) {
    # Start-Process: Windows PowerShell turns the stderr of a native program into an error
    $p = Start-Process -FilePath $test -Wait -PassThru -NoNewWindow `
            -RedirectStandardOutput (Join-Path $work "test.out") -RedirectStandardError (Join-Path $work "test.log")
    Report "unit test" ($p.ExitCode -eq 0) "test-mul-mat-id-skip"
} else {
    Report "unit test" $false "$test not found, build with -DLLAMA_BUILD_TESTS=ON"
}

# 2. perplexity on the docs of the repository, ubatch 16 so that the hybrid path is used
$corpus = Join-Path $work "corpus.txt"
Get-ChildItem (Join-Path $PSScriptRoot "..\docs") -Recurse -Filter *.md | Sort-Object FullName | Get-Content -Raw |
    Set-Content -Encoding utf8 $corpus
$ppl_args = @("-f", $corpus, "-c", "512", "-b", "512", "-ub", "16", "--chunks", "4")
$ppl_base   = Get-Ppl (Invoke-Tool "llama-perplexity" ($common + $base   + $ppl_args) "ppl-base")
$ppl_hybrid = Get-Ppl (Invoke-Tool "llama-perplexity" ($common + $hybrid + $ppl_args) "ppl-hybrid")
$diff = [math]::Abs($ppl_hybrid - $ppl_base) / $ppl_base
Report "perplexity" ($diff -le $MaxPplDiff) ("baseline {0:N4}, hybrid {1:N4}, diff {2:P2}" -f $ppl_base, $ppl_hybrid, $diff)

# 3. determinism of the hybrid path (CPU and GPU run at the same time)
$prompt = "Write a detailed technical article about how mixture-of-experts language models are served on consumer hardware."
$gen = @("-p", $prompt, "--temp", "0", "--top-k", "1", "--seed", "1", "-no-cnv", "--no-display-prompt", "--ignore-eos")
$null = Invoke-Tool "llama-completion" ($common + $hybrid + $gen + @("-n", "128")) "det-1"
$null = Invoke-Tool "llama-completion" ($common + $hybrid + $gen + @("-n", "128")) "det-2"
$same = (Get-FileHash (Join-Path $work "det-1.out")).Hash -eq (Get-FileHash (Join-Path $work "det-2.out")).Hash
Report "determinism" $same "two hybrid runs, 128 tokens"

# 4. speed
$tps_base   = Get-Tps (Invoke-Tool "llama-completion" ($common + $base   + $gen + @("-n", "512")) "speed-base")
$tps_hybrid = Get-Tps (Invoke-Tool "llama-completion" ($common + $hybrid + $gen + @("-n", "512")) "speed-hybrid")
$msg = "baseline {0:N1} t/s, hybrid {1:N1} t/s ({2:+0;-0}%)" -f $tps_base, $tps_hybrid, (100*($tps_hybrid/$tps_base - 1))
if ($tps_hybrid -ge $tps_base) { Write-Host ("{0,-12} {1,-5} {2}" -f "speed", "OK", $msg) } else { Write-Host ("{0,-12} {1,-5} {2}" -f "speed", "WARN", $msg) }

Write-Host ""
Write-Host $(if ($failed -eq 0) { "all checks passed (logs in $work)" } else { "$failed check(s) failed (logs in $work)" })
exit $failed
