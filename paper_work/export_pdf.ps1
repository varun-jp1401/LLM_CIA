param([string]$src, [string]$dst)
$w = New-Object -ComObject Word.Application; $w.Visible = $false
try {
  $d = $w.Documents.Open($src)
  $d.Fields.Update() | Out-Null
  $d.Repaginate()
  $d.ExportAsFixedFormat($dst, 17)
  "pages: " + $d.ComputeStatistics(2) + "  words: " + $d.ComputeStatistics(0)
  $d.Close(0)
} catch { $_.Exception.Message } finally { $w.Quit() }
