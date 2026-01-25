# Фаза 4 — Runtime и оркестрация (батчинг, QoS, метрики)

## Цель
Обеспечить стабильный потоковый пайплайн с очередями, батчингом, QoS, логами и метриками.

## Epics → Stories → Subtasks
- Epic: Очереди и батчинг
  - Story: Батчер
    - Subtask: Микро-окно (5–20 мс), лимиты по токенам/памяти.
    - Subtask: Приоритеты: streaming > batch; сортировка по длине.
  - Story: QoS
    - Subtask: Дедлайны/тайм-ауты; отмена запросов.
    - Subtask: Лимит одновременных стримов; backpressure.
- Epic: Буфер аудио и гладкость
  - Story: Целевой буфер 200–500 мс, защита от underrun/overrun.
  - Story: Tail padding, корректное завершение.
- Epic: Логи и метрики
  - Story: Structured logs (tracing JSON)
    - Subtask: trace_id/session_id/req_id, stage markers.
  - Story: Метрики
    - Subtask: latency p50/p95, t2fa, RTF, batch size, drop/timeout rate.
    - Subtask: Prometheus exporter или sink.
- Epic: Конфигурирование
  - Story: `runtime.toml`
    - Subtask: Очереди, батч-окна, лимиты памяти/KV cache, device policy.

## Deliverables
- Крейты `runtime` с батчером, QoS и метриками.
- Документация по настройке `runtime.toml`.
- Интеграция логов/метрик в пайплайн.

## DoD
- Потоковый запрос проходит end-to-end без underrun/overrun.
- Метрики и логи собираются; p95 t2fa и RTF отображаются.

## Риски
- Неправильная эвикция KV cache → OOM или деградация; смягчение — лимиты и мониторинг.
- Адаптивный батч может увеличивать latency для коротких запросов; смягчение — приоритеты/микро-окно.

## Метрики проверки
- p95 time-to-first-audio в целевых пределах (см. KPI).
- Drop/timeout rate ≈ 0 на нагрузочном тесте.
