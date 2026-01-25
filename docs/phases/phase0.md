# Фаза 0 — Подготовка и инфраструктура

## Цель
Создать основу для разработки: конфиги, схемы весов, CI-черновик, метрики/логи-шаблоны, минимальные бенчи.

## Epics → Stories → Subtasks
- Epic: Базовая инфраструктура
  - Story: Структура workspace
    - Subtask: Описать crates в README/workspace, добавить `Cargo.toml` заготовки.
    - Subtask: Настроить feature-flags (cpu, cuda, server, bench).
  - Story: Конфиги моделей
    - Subtask: Определить schema `model.toml` (dims, vocab, codec, backend).
    - Subtask: Определить `runtime.toml` и `server.toml` черновик.
  - Story: Загрузка весов
    - Subtask: Документировать требования к `safetensors` именованию параметров.
    - Subtask: Определить checksums/версионирование весов.
  - Story: CI черновой
    - Subtask: Шаблон `fmt + clippy + tests` (без heavy весов).
    - Subtask: Заглушки для мини-весов в CI (skip если нет).
  - Story: Логи/метрики базовые
    - Subtask: Подключить `tracing` JSON формат (конфиг по env).
    - Subtask: Шаблон метрик (Prometheus exporter интерфейс).
  - Story: Бенчи-шаблон
    - Subtask: Настроить `criterion` каркас без реальных моделей (mock ops).

## Deliverables
- `docs/roadmap.md`, `docs/architecture.md`, план фаз.
- Черновики `model.toml`, `runtime.toml`, `server.toml` (образцы).
- CI конфиг (заглушка) + инструкции к мини-весам.
- Шаблон бенчей и логирования.

## DoD
- Workspace и файлы конфигов описаны и валидируются (schema check).
- CI прогоняет `fmt`, `clippy`, unit-заглушки.
- Есть мок-бенч, логи в JSON выводятся.

## Риски
- Формат весов не финализирован → держать версионность и backward-совместимость.
- CI может падать из-за отсутствия весов → маркировать тесты/бенчи feature-флагами.

## Метрики проверки
- Успешный прогон CI без весов.
- Валидация config-схем (toml/jsonschema) проходит.
