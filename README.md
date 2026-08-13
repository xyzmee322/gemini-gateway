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

## OpenRouter для embeddings

Gateway может использовать OpenRouter без proxy для `POST /v1/embeddings` и только для модели `google/gemini-embedding-2`.

Оба режима выключены по умолчанию, чтобы случайно не включить платный путь.

Direct-only режим сразу отправляет `google/gemini-embedding-2` в OpenRouter по `GEMINI_GATEWAY_OPENROUTER_API_KEY`, без Gemini routes и proxy:

В этом режиме одинаково обрабатываются текст, изображения и `inline_data` с изображениями, аудио, видео или файлами. Репозиторий Keyproxy для embedding-запросов не вызывается; аудио передаётся в OpenRouter напрямую.

```powershell
$env:GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_DIRECT_ONLY_ENABLED="true"
$env:GEMINI_GATEWAY_OPENROUTER_API_KEY="sk-or-..."
```

Fallback режим сначала пробует Gemini routes, а OpenRouter включает только после ошибок маршрутов:

```powershell
$env:GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED="true"
$env:GEMINI_GATEWAY_OPENROUTER_API_KEY="sk-or-..."
$env:GEMINI_GATEWAY_MAX_ROUTE_ATTEMPTS="5"
```

Условия fallback:

- сначала gateway пытается выполнить запрос через Gemini routes `api_key + proxy`;
- при retryable provider/transport ошибках gateway берёт следующий route до `GEMINI_GATEWAY_MAX_ROUTE_ATTEMPTS`;
- OpenRouter вызывается для embeddings только после route acquisition failure или исчерпания Gemini routes для `google/gemini-embedding-2`;
- разрешённые причины включают `no_route`, `cooldown_active`, `quota_exhausted`, `network_timeout`, `proxy_failed`, `rate_limited`, `provider_unavailable`;
- chat completions и TTS никогда не используют OpenRouter fallback.

В direct-only ответе route metadata будет `transport_mode: direct`, `project_label: openrouter-direct`, `route_label: openrouter-embedding-direct`.
В fallback ответе route metadata будет `transport_mode: direct`, `project_label: openrouter-fallback`, `route_label: openrouter-embedding-fallback`.

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
