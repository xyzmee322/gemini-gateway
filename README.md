# Gemini Gateway

Отдельный proxy-only сервис для Gemini: выбирает связку `api_key + proxy`, учитывает rate limits, cooldowns, retries и возвращает совместимый внутренний HTTP-контракт для chat, embeddings и TTS.

## Контракт

- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `POST /v1/audio/speech`
- `GET /health/live`
- `GET /health`

Авторизация: `Authorization: Bearer <GEMINI_GATEWAY_TOKEN>`.

## Локальный запуск

```powershell
docker compose up -d postgres migrations gateway
```

Seed маршрутов запускается отдельно, когда заданы `GEMINI_API_KEY` и `GEMINI_GATEWAY_PROXY_URL`:

```powershell
docker compose --profile seed run --rm seed
```

## Перенос данных из Soybob V3

`scripts/clone_gateway_data.sql` рассчитан на сценарий, где старая схема `soybob_v3` и новая схема `gemini_gateway` доступны в одной Postgres-базе. Сначала разверни миграции нового сервиса, затем останови старый gateway traffic и скопируй данные:

```powershell
docker compose up -d postgres migrations
.\scripts\clone_gateway_data.ps1
```

Скрипт копирует таблицы из `soybob_v3` в `gemini_gateway` с сохранением `id`. Это важно: `cooldowns.scope_key` хранит id route-сущностей как текст.

Нужно использовать те же `GEMINI_GATEWAY_ENCRYPTION_KEY` и `GEMINI_GATEWAY_HMAC_KEY`, иначе старые encrypted API keys/proxy credentials не расшифруются.

Если целевой gateway использует физически отдельную БД, сначала перенеси данные через `pg_dump`/`pg_restore` или временно подключи новый сервис к общей базе для clone-шага.

## Тесты

```powershell
python -m pytest tests/gemini_gateway -q
```
