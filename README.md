# САВОС — сервер инференса Triton

Репозиторий инференса для проекта **САВОС**: акустическое восприятие для автономных транспортных средств. Восьмиканальный WAV с круговой микрофонной решетки подается в конвейер, развернутый через Triton, который:

1. **Локализует** источник звука: грубая GCC-PHAT-оценка по парам «два квадрата» из ТЗ, затем SRP-PHAT-уточнение в секторе +/-5°
2. **Выбирает** микрофон, ближайший к оцененному направлению прихода сигнала
3. **Извлекает** log-Mel-спектрограмму через librosa для CNN моделей
4. **Классифицирует** событие выбранным в GUI CNN- или AST-классификатором
5. Для сирен спецтранспорта **оценивает** расстояние по пиковой амплитуде по ТЗ

## Классы (17)

car_acceleration · car_braking · car_horn · car_idling · moto_acceleration · moto_idling · siren_1 · siren_4 · siren_5 · tram · tram_acceleration · tram_braking · tram_ring · truck_acceleration · truck_braking · truck_horn · truck_idling

## Быстрый старт

```bash
# 1. Репозиторий настроен на поставляемые ONNX-классификаторы:
cp classification-models/furletov_cnn_baseline_tuned.onnx model_repository/classifier/1/model.onnx
cp classification-models/furletov_ast_finetune_tuned.onnx model_repository/classifier_furletov_ast/1/model.onnx
cp classification-models/us8k_cnn_baseline_tuned.onnx model_repository/classifier_us8k_cnn/1/model.onnx
cp classification-models/us8k_ast_finetune_tuned.onnx model_repository/classifier_us8k_ast/1/model.onnx

# 2. labels.json: список классов должен совпадать с порядком логитов model.onnx,
#    а `input_name` / `output_name` должны совпадать с ONNX-именами входов и выходов.
$EDITOR model_repository/classifier/labels.json

# 3. (Опционально) Сгенерируйте демонстрационный 8-канальный WAV
python scripts/generate_demo_asset.py --doa 75 --out demo/assets/example_8ch.wav

# 4. Запустите все сервисы
docker compose up --build
```

Затем откройте http://localhost:7860 и загрузите 8-канальный WAV.

## Адреса сервисов

| Сервис         | URL                                   |
|----------------|---------------------------------------|
| Gradio UI      | http://localhost:7860                 |
| Triton HTTP    | http://localhost:8000/v2              |
| Triton gRPC    | http://localhost:8001                 |
| Triton metrics | http://localhost:8002/metrics         |

## Проверка через CLI

```bash
docker compose exec gradio python -m demo.client --wav demo/assets/example_8ch.wav
```

Команда выводит JSON с полями `classifier_model`, `class_name`, `confidence`, `doa_deg`, `selected_mic`, `distance_m`, `is_emv`.

## Структура репозитория

```
model_repository/
├── pipeline/            # BLS-оркестратор: единственная модель, к которой обращается клиент
├── localizer/           # Python: GCC-PHAT + SRP-PHAT
├── channel_selector/    # Python: ближайший микрофон по DOA
├── feature_extractor/   # Python: log-Mel через librosa
├── ast_feature_extractor/ # Python: input_values для AST через transformers/torchaudio
├── classifier/          # Furletov CNN по умолчанию на ONNX Runtime backend
│   ├── 1/model.onnx     # чекпойнт
│   ├── labels.json      # классы по умолчанию, EmV-подмножество, ONNX I/O-имена
│   └── models.json      # каталог классификаторов для GUI/конвейера
├── classifier_furletov_ast/
├── classifier_us8k_ast/
└── classifier_us8k_cnn/ # Опциональный CNN-классификатор UrbanSound8K
demo/
├── app.py               # Gradio UI
├── client.py            # CLI-клиент
└── assets/              # демонстрационные WAV-файлы
docker/                  # Dockerfile.triton, Dockerfile.gradio
docker-compose.yml
scripts/                 # разовые вспомогательные скрипты: dummy ONNX, синтез WAV
tests/                   # unit-тесты pytest
```

## Соглашение об индексации микрофонов

Канал `i` (0..7) в WAV соответствует микрофону с азимутом `i·45°` против часовой стрелки от продольной оси автомобиля вперед (+X, 0°). Микрофон 0 направлен вперед, микрофон 2 — влево, микрофон 4 — назад, микрофон 6 — вправо.

## Соглашения и допущения

- Интегрированы классификаторы с выбором в GUI: `furletov_cnn`, `furletov_ast`, `us8k_cnn` и `us8k_ast`.
- Вход CNN-классификатора: `mel_spectrogram`, `FP32[B, T, 128]` (batch, time, n_mels). Временная ось динамическая.
- Вход AST-классификатора: `input_values`, `FP32[B, 1024, 128]`.
- Выход ONNX-классификатора: логиты `FP32[B, C]`, где `C` зависит от выбранного классификатора.
- Mel-параметры задаются в `model_repository/feature_extractor/config.pbtxt`: `target_sr=16000, target_sec=10.0, n_mels=128, n_fft=1024, hop_length=512`.
- AST-параметры задаются в `model_repository/ast_feature_extractor/config.pbtxt`: `target_sr=16000, max_length=1024, num_mel_bins=128, mean=-4.2677393, std=4.5689974`.
- Имена ONNX I/O-тензоров и списки классов для выбора в GUI находятся в `model_repository/classifier/models.json`.
- Константы калибровки расстояния `A0`, `R0` находятся в `parameters` файла `model_repository/pipeline/config.pbtxt` и требуют эмпирической калибровки.

## Разработка

```bash
pip install -r requirements.txt
pytest tests/                                         # unit-тесты, Triton не нужен
python scripts/export_dummy_classifier.py             # опциональная замена классификатора на совместимый небольшой random ONNX
python scripts/generate_demo_asset.py --doa 75        # сборка синтетического 8-канального WAV
docker compose up --build                             # полный стек
```
