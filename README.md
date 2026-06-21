# Gemini Gateway

Отдельный proxy-only сервис для Gemini: выбирает связку `api_key + proxy`, учитывает rate limits, cooldowns, retries и возвращает совместимый внутренний HTTP-контракт для chat, embeddings и TTS.

## Контракт

- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `POST /v1/audio/speech`
- `GET /health/live`
- `GET /health`

Авторизация: `Authorization: Bearer <GEMINI_GATEWAY_TOKEN>`.

## Мониторинг proxy

- `GET /admin/monitor` отдаёт HTML-дашборд без bearer-авторизации на саму страницу.
- JSON API остаётся закрытым bearer-токеном: `GET /admin/monitor/api/summary` и `GET /admin/monitor/api/timeseries`.
- В форме дашборда укажи token, minutes, bucket_seconds, model, proxy_label и refresh_seconds. Токен хранится локально в браузере и отправляется только в заголовке `Authorization: Bearer <token>`.
- При включённом auto дашборд сам обновляет summary и timeseries, поэтому его можно держать открытым как real-time экран состояния proxy.

Как читать latency stages:

- высокий `response_headers_ms` означает, что запрос уже ушёл к провайдеру через proxy, и gateway ждёт первые headers;
- высокий `response_body_ms` означает, что headers получены, но тело ответа читается медленно;
- `timeout_kind=write_timeout` указывает на проблему отправки тела запроса;
- `timeout_kind=read_timeout` вместе с `timeout_stage=response_headers` означает ожидание первого ответа от провайдера;
- `payload_kind=media` отделяет медиа-запросы от текстовых, чтобы не смешивать разные профили latency.

## Локальный запуск

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres migrations gateway
```

Seed маршрутов запускается отдельно, когда заданы `GEMINI_API_KEY` и `GEMINI_GATEWAY_PROXY_URL`:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile seed run --rm seed
```

## OpenRouter fallback для embeddings

Gateway может использовать OpenRouter без proxy только как fallback для `POST /v1/embeddings` и только для модели `google/gemini-embedding-2`.

Fallback выключен по умолчанию, чтобы случайно не включить платный путь. Для включения:

```powershell
$env:GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED="true"
$env:GEMINI_GATEWAY_OPENROUTER_API_KEY="sk-or-..."
```

Условия срабатывания:

- сначала gateway пытается выдать обычный Gemini route `api_key + proxy`;
- OpenRouter вызывается только если route pool не может выдать ни одного Gemini route для `google/gemini-embedding-2`;
- разрешённые причины: `no_route`, `cooldown_active`, `quota_exhausted`;
- fallback не срабатывает после ошибки одного уже выбранного Gemini route, потому что это не доказывает недоступность всех ключей;
- chat completions и TTS никогда не используют OpenRouter fallback.

В ответе route metadata будет `transport_mode: direct`, `project_label: openrouter-fallback`, `route_label: openrouter-embedding-fallback`.

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

## Операции

- [Zombie process monitoring](docs/operations/zombie-processes.md): проверка `[python] <defunct>`, `init: true` и post-deploy чеклист.
