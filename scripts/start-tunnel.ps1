# start-tunnel.ps1
# Starts cloudflared quick tunnel and auto-registers the Telegram webhook with the new URL.
# Re-run this every time cloudflared restarts (ephemeral URLs change on each run).

param(
    [string]$LocalUrl = "http://localhost:8000",
    [string]$EnvFile  = "$PSScriptRoot\..\.env"  # moved from repo root into scripts/
)

# Parse .env
$cfg = @{}
Get-Content $EnvFile | Where-Object { $_ -match '^[^#]' -and $_ -match '=' } | ForEach-Object {
    $k, $v = $_ -split '=', 2
    $cfg[$k.Trim()] = $v.Trim()
}

$token  = $cfg['TELEGRAM_BOT_TOKEN']
$secret = $cfg['WEBHOOK_SECRET']
$slug   = if ($cfg['TENANT_SLUG']) { $cfg['TENANT_SLUG'] } else { 'demo' }

if (-not $token)  { Write-Error "TELEGRAM_BOT_TOKEN not in .env"; exit 1 }
if (-not $secret) { Write-Error "WEBHOOK_SECRET not in .env"; exit 1 }

$log = "$env:TEMP\cloudflared-$(Get-Date -f 'yyyyMMddHHmmss').log"

Write-Host "Starting cloudflared → $LocalUrl"
$proc = Start-Process cloudflared `
    -ArgumentList "tunnel", "--url", $LocalUrl `
    -RedirectStandardError $log `
    -NoNewWindow -PassThru

# Poll stderr log until cloudflared prints its trycloudflare.com URL (~5s)
$cfUrl   = $null
$deadline = [datetime]::Now.AddSeconds(30)
while ([datetime]::Now -lt $deadline -and -not $cfUrl) {
    Start-Sleep -Milliseconds 500
    $content = if (Test-Path $log) { Get-Content $log -Raw -EA SilentlyContinue } else { "" }
    if ($content -match 'https://[a-z0-9-]+\.trycloudflare\.com') {
        $cfUrl = $Matches[0]
    }
}

if (-not $cfUrl) {
    if (Test-Path $log) { Get-Content $log }
    Write-Error "Timed out waiting for cloudflared URL (30s)"
    Stop-Process -Id $proc.Id -Force
    exit 1
}

Write-Host "Tunnel URL: $cfUrl"

# Register Telegram webhook
$webhookUrl = "$cfUrl/webhook/telegram/$slug"
$body = @{ url = $webhookUrl; secret_token = $secret } | ConvertTo-Json
$resp = Invoke-RestMethod "https://api.telegram.org/bot$token/setWebhook" `
    -Method Post -ContentType "application/json" -Body $body

Write-Host "setWebhook: $($resp.description)"
Write-Host "Active webhook: $webhookUrl"
Write-Host ""
Write-Host "Ctrl+C to stop tunnel."

try {
    Get-Content $log -Wait
} finally {
    Write-Host "Stopping cloudflared (PID $($proc.Id))..."
    Stop-Process -Id $proc.Id -Force -EA SilentlyContinue
}
