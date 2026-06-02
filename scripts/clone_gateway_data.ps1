param(
    [string]$ComposeFile = "docker-compose.yml",
    [string]$PostgresService = "postgres",
    [string]$PostgresUser = $env:POSTGRES_USER,
    [string]$PostgresDb = $env:POSTGRES_DB
)

if (-not $PostgresUser) {
    $PostgresUser = "gemini_gateway"
}

if (-not $PostgresDb) {
    $PostgresDb = "gemini_gateway"
}

$scriptPath = Join-Path $PSScriptRoot "clone_gateway_data.sql"
if (-not (Test-Path $scriptPath)) {
    throw "clone_gateway_data.sql not found"
}

Get-Content -Encoding UTF8 $scriptPath | docker compose -f $ComposeFile exec -T $PostgresService psql -v ON_ERROR_STOP=1 -U $PostgresUser -d $PostgresDb
