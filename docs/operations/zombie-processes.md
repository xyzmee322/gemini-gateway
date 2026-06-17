# Zombie Process Monitoring

## Симптом

У gateway-контейнера могут накапливаться zombie-процессы вида `[python] <defunct>` с PPID основного `uvicorn` процесса. Если приложение запущено как PID 1 без init-reaper, orphan child-процессы не всегда забираются через `wait()`.

## Текущий фикс

`gateway` service запускается с `init: true`, чтобы Docker добавлял tiny init/reaper. Healthcheck использует exec-form `CMD`, а не `CMD-SHELL`, чтобы не плодить shell-wrapper перед `python -c`. Ошибки healthcheck завершаются тихим `exit 1`, без Python traceback в Docker health log.

## Проверка на хосте

```bash
ps -eo stat,ppid,comm,args | awk '$1 ~ /^Z/ {count++; byppid[$2]++} END {print "zombies", count+0; for (ppid in byppid) print ppid, byppid[ppid]}'
```

Норма: `zombies 0`.

Если zombie есть, сопоставь PPID с gateway-контейнером:

```bash
docker inspect --format '{{.State.Pid}}' gemini-gateway-gateway-1
ps -o pid,ppid,stat,comm,args -p <gateway_host_pid>
ps -o pid,ppid,stat,comm,args --ppid <gateway_host_pid>
```

## Проверка после деплоя

1. Пересоздать `gemini-gateway-gateway-1`, чтобы применился `init: true`.
2. Залить пачку картинок через pic-graph upload worker.
3. Подождать 30-60 минут.
4. Повторить zombie-check и убедиться, что счетчик не растет.
5. Проверить `docker stats gemini-gateway-gateway-1`.
6. Проверить логи gateway на ошибки healthcheck, subprocess timeout и сетевые timeout.

## Временный workaround

```bash
docker restart gemini-gateway-gateway-1
```

Это чистит текущие zombie вместе с родительским процессом, но без `init: true` не устраняет причину накопления.
