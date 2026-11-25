# Generating synthetic datasets

All commands below are run from `ai-service\datasets` using the service virtual environment (`..\venv\Scripts\python.exe`).

## 1. Install dependencies

If not already installed in `ai-service\venv`:

```powershell
cd ai-service
.\venv\Scripts\pip install -r requirements.txt
```

## 2. Generate a full 7-day `lights` dataset from `lights_schedule.md`

Timestamps passed to `--start` are interpreted in **`Asia/Manila` (`+08:00`)** when naive (or converted to `Asia/Manila` when tz-aware) and stored in UTC ISO-8601 (`...Z`) matching [Receiver.ino](../../firmware/Receiver/Receiver.ino) and [PZEM004T.ino](../../firmware/PZEM004T/PZEM004T.ino).

### Overwrite daily CSV files (`Monday_dataset.csv` … `Sunday_dataset.csv`) and write Week 1 to `test_data.db`

```powershell
cd ai-service\datasets
..\venv\Scripts\python.exe ..\scripts\generate_dataset.py `
  --schedule ..\docs\lights_schedule.md `
  --start "2026-10-05T00:00:00+08:00" `
  --duration-hours 168 `
  --interval-seconds 5 `
  --agent "B1-GF-Room1" `
  --load lights `
  --seed 42 `
  --daily-csv-dir . `
  --mode sqlite `
  --database "..\..\backend\test_data.db" `
  --replace
```

### Append Week 2 (held-out validation week) into `test_data.db`

```powershell
..\venv\Scripts\python.exe ..\scripts\generate_dataset.py `
  --schedule ..\docs\lights_schedule.md `
  --start "2026-10-12T00:00:00+08:00" `
  --duration-hours 168 `
  --interval-seconds 5 `
  --agent "B1-GF-Room1" `
  --load lights `
  --seed 1042 `
  --energy-start 1.25 `
  --mode sqlite `
  --database "..\..\backend\test_data.db"
```

Key flags:
- `--schedule`: parses ON windows per weekday from [lights_schedule.md](../docs/lights_schedule.md) in `Asia/Manila`.
- `--morning-bulbs 2`: half open (~9.5 W) during morning blocks (`< 12:00` Manila).
- `--afternoon-bulbs 3,4`: 3–4 bulbs (~13.5–19.6 W) during afternoon/evening blocks (`>= 12:00` Manila).
- `--voltage-range 215.0:235.0`: normal Philippine mains voltage envelope inside the `207–253 V` safe band.
- `--load lights --replace`: writes directly into `load_readings` (`load = 'lights'`) and creates `lights_readings` view without touching `energy_data`.

## 3. Train the per-load `lights` model

```powershell
cd ai-service
.\venv\Scripts\python.exe train_from_db.py --db test_data.db --load lights
```

This writes exclusively to `ai-service\models\lights\`.
