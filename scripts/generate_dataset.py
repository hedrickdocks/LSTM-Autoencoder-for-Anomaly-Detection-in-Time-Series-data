import argparse
import json
import os
import re
import sqlite3
import urllib.request
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    'datetime',
    'agent',
    'voltage',
    'current',
    'power',
    'energy',
    'power_factor',
]
DATABASE_TIMEZONE = ZoneInfo('Asia/Manila')
MEASUREMENT_COLUMNS = ['voltage', 'current', 'power', 'energy', 'power_factor']
MEASUREMENT_DECIMALS = 3

DAY_NAME_MAP = {
    'monday': 0,
    'tuesday': 1,
    'wednesday': 2,
    'wednsesday': 2,  # tolerate legacy spelling
    'thursday': 3,
    'friday': 4,
    'saturday': 5,
    'sunday': 6,
}
DAY_INDEX_TO_NAME = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday',
]
LOAD_VIEWS = {
    'lights': 'lights_readings',
    'fan': 'ceiling_fan_readings',
    'outlet': 'outlet_readings',
    'aircon': 'aircon_readings',
}


def format_storage_datetime(value):
    return value.astimezone(ZoneInfo('UTC')).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def _parse_hhmm(token):
    token = token.strip()
    if ':' in token:
        h_str, m_str = token.split(':', 1)
        return int(h_str) * 60 + int(m_str)
    return int(token) * 60


def parse_schedule_markdown(path):
    """Parse weekly ON blocks (minutes from midnight, Asia/Manila) from lights_schedule.md."""
    schedule = {day: [] for day in range(7)}
    current_day = None
    range_pattern = re.compile(r'^(\d{1,2}(?::\d{2})?)\s*[-–]\s*(\d{1,2}(?::\d{2})?)\b')

    with open(path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            low = line.lower()
            if low in DAY_NAME_MAP:
                current_day = DAY_NAME_MAP[low]
                continue
            if current_day is None:
                continue
            if low == 'off':
                continue
            match = range_pattern.match(line)
            if match:
                start_min = _parse_hhmm(match.group(1))
                end_min = _parse_hhmm(match.group(2))
                if 0 <= start_min < end_min <= 1440:
                    schedule[current_day].append((start_min, end_min))

    for day in schedule:
        schedule[day].sort()
    return schedule


def parse_break_times(break_times_str):
    if not break_times_str:
        return []

    ranges = []
    for time_range in break_times_str.split(','):
        start_str, end_str = time_range.strip().split('-')
        start_time = datetime.strptime(start_str.strip(), '%H:%M').time()
        end_time = datetime.strptime(end_str.strip(), '%H:%M').time()
        ranges.append((start_time, end_time))
    return ranges


def parse_range_pair(spec):
    lo_str, hi_str = spec.split(':', 1)
    lo, hi = float(lo_str), float(hi_str)
    if hi < lo:
        raise ValueError(f'Invalid range {spec}: max < min')
    return lo, hi


def parse_bulb_choices(spec):
    choices = [int(x.strip()) for x in spec.split(',') if x.strip()]
    if not choices or any(c < 0 for c in choices):
        raise ValueError(f'Invalid bulb choice list: {spec}')
    return choices


def parse_stage_watts(spec):
    stages = [float(x.strip()) for x in spec.split(',') if x.strip()]
    if not stages or any(w <= 0 for w in stages):
        raise ValueError(f'Invalid unit stage wattages: {spec}')
    return tuple(stages)


def load_source(path, agent=None, load=None):
    if path.lower().endswith(('.db', '.sqlite', '.sqlite3')):
        connection = sqlite3.connect(path)
        tables = {
            r[0]
            for r in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if load and 'load_readings' in tables:
            query = 'SELECT datetime, agent, voltage, current, power, energy, power_factor FROM load_readings WHERE load = ?'
            params = [load]
            if agent:
                query += ' AND agent = ?'
                params.append(agent)
        else:
            query = 'SELECT datetime, agent, voltage, current, power, energy, power_factor FROM energy_data'
            params = []
            if agent:
                query += ' WHERE agent = ?'
                params.append(agent)
        data = pd.read_sql_query(query, connection, params=params, parse_dates=['datetime'])
        connection.close()
    else:
        data = pd.read_csv(path, parse_dates=['datetime'])

    missing = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f'Source is missing columns: {sorted(missing)}')

    data = data[REQUIRED_COLUMNS].copy()
    data['datetime'] = pd.to_datetime(data['datetime'], utc=True, errors='coerce')
    data = data.dropna(subset=['datetime']).sort_values('datetime')
    for column in MEASUREMENT_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    data = data.dropna(subset=MEASUREMENT_COLUMNS)
    if data.empty:
        raise ValueError('Source does not contain usable telemetry rows.')
    return data


def mean_reverting_series(rng, minimum, maximum, count, rho=0.994, step_scale=0.055, start_value=None):
    """Generate a smooth bounded Ornstein-Uhlenbeck-like series across [minimum, maximum]."""
    if maximum <= minimum:
        return np.full(count, minimum, dtype=float)

    span = maximum - minimum
    midpoint = 0.5 * (minimum + maximum)
    sigma = span * step_scale
    noise = rng.normal(0.0, sigma, size=count)

    values = np.empty(count, dtype=float)
    values[0] = start_value if start_value is not None else rng.uniform(minimum, maximum)
    for i in range(1, count):
        nxt = midpoint + rho * (values[i - 1] - midpoint) + noise[i]
        values[i] = min(maximum, max(minimum, nxt))
    return values


def _merge_contiguous_intervals(intervals):
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for start_m, end_m in intervals[1:]:
        if start_m <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end_m)
        else:
            merged.append([start_m, end_m])
    return [(s, e) for s, e in merged]


def build_schedule_bulb_profile(
    timestamps_manila,
    schedule,
    rng,
    morning_bulbs,
    afternoon_bulbs,
    jitter_minutes=2.0,
    interval_seconds=5,
    bulb_watts=4.75,
    unit_stages=None,
):
    """Build per-step active unit count m[t] and nominal per-unit watts w[t] from Asia/Manila schedule."""
    count = len(timestamps_manila)
    bulbs = np.zeros(count, dtype=int)
    stages = tuple(unit_stages) if unit_stages else (float(bulb_watts),)
    unit_watts = np.full(count, stages[0], dtype=float)

    local_dates = timestamps_manila.date
    minute_of_day = (
        timestamps_manila.hour * 60.0
        + timestamps_manila.minute
        + timestamps_manila.second / 60.0
    )

    unique_dates = pd.Index(local_dates).unique()
    for day_date in unique_dates:
        dow = day_date.weekday()
        raw_blocks = schedule.get(dow, [])
        if not raw_blocks:
            continue

        day_mask = local_dates == day_date
        day_indices = np.flatnonzero(day_mask)
        day_mins = minute_of_day[day_mask]

        merged = _merge_contiguous_intervals(raw_blocks)
        for seg_start, seg_end in merged:
            s_jit = seg_start + (rng.uniform(-jitter_minutes, jitter_minutes) if jitter_minutes > 0 else 0.0)
            e_jit = seg_end + (rng.uniform(-jitter_minutes, jitter_minutes) if jitter_minutes > 0 else 0.0)
            s_jit = max(0.0, s_jit)
            e_jit = min(1440.0, max(s_jit + 1.0, e_jit))

            split_min = 720.0 + (rng.uniform(0.0, jitter_minutes) if jitter_minutes > 0 else 0.0)

            in_seg = (day_mins >= s_jit) & (day_mins < e_jit)
            seg_pos = np.flatnonzero(in_seg)
            if len(seg_pos) == 0:
                continue

            seg_mins = day_mins[seg_pos]
            morn_mask = seg_mins < split_min
            aft_mask = ~morn_mask

            seg_bulbs = np.zeros(len(seg_pos), dtype=int)
            seg_watts = np.full(len(seg_pos), stages[0], dtype=float)

            for sub_mask, unit_pool in ((morn_mask, morning_bulbs), (aft_mask, afternoon_bulbs)):
                sub_pos = np.flatnonzero(sub_mask)
                if len(sub_pos) == 0:
                    continue
                n_sub = len(sub_pos)
                # Multi-stage or multi-unit loads step cleanly across levels/speeds inside long blocks
                if len(stages) > 1 or len(unit_pool) > 1:
                    min_seg_len = max(30, int(1800 / interval_seconds))  # >= 30 min per stage when possible
                    max_cuts = max(1, min(4, n_sub // max(1, min_seg_len)))
                    n_chunks = int(rng.integers(1, max_cuts + 1)) if max_cuts > 1 else 1
                    boundaries = [0]
                    if n_chunks > 1 and n_sub > n_chunks * 24:
                        cuts = sorted(rng.choice(np.arange(24, n_sub - 24), size=n_chunks - 1, replace=False).tolist())
                        boundaries.extend(cuts)
                    boundaries.append(n_sub)
                    for b0, b1 in zip(boundaries[:-1], boundaries[1:]):
                        seg_bulbs[sub_pos[b0:b1]] = int(rng.choice(unit_pool))
                        seg_watts[sub_pos[b0:b1]] = float(rng.choice(stages))
                else:
                    seg_bulbs[sub_pos] = int(unit_pool[0])
                    seg_watts[sub_pos] = float(rng.choice(stages))

            # Realistic switch stagger at turn-on (~50% of turn-ons spend 1-2 packets at 1 unit)
            if seg_bulbs[0] >= 2 and len(seg_bulbs) > 4 and rng.random() < 0.5:
                stagger_steps = int(rng.integers(1, 3))
                seg_bulbs[:stagger_steps] = 1

            bulbs[day_indices[seg_pos]] = seg_bulbs
            unit_watts[day_indices[seg_pos]] = seg_watts

    return bulbs, unit_watts


def synthesize_pzem_window_telemetry(
    timestamps_manila,
    bulbs,
    rng,
    voltage_range,
    pf_range,
    bulb_watts,
    energy_start,
    interval_seconds=5,
    subsamples=18,
    agent='B1-GF-Room1',
    unit_watts=None,
    off_current_max=0.0,
    multi_unit_pf_min=None,
):
    """
    Synthesize 5-second PZEM-004T window telemetry matching firmware/PZEM004T/PZEM004T.ino:
      - Sub-samples polled every ~250ms (N=18), quantized to PZEM resolutions
        (0.1 V, 0.001 A, 0.1 W, 0.01 PF, 0.001 kWh).
      - OFF state has power = 0.0, power_factor = 0.0 (and optional standby CT current noise).
      - Transition packets simulate mid-window switching at a random sub-sample cut k*.
      - Window outputs are arithmetic means over the N sub-samples (lines 518-526).
    """
    count = len(timestamps_manila)
    v_min, v_max = voltage_range
    pf_min, pf_max = pf_range

    diurnal_seconds = (
        timestamps_manila.hour.to_numpy(dtype=float) * 3600.0
        + timestamps_manila.minute.to_numpy(dtype=float) * 60.0
        + timestamps_manila.second.to_numpy(dtype=float)
    )
    diurnal_phase = 2.0 * np.pi * diurnal_seconds / 86400.0
    v_diurnal = -0.15 * (v_max - v_min) * np.sin(diurnal_phase)
    v_base = mean_reverting_series(rng, v_min, v_max, count, rho=0.996, step_scale=0.05)
    v_base = np.asarray(np.clip(v_base + v_diurnal, v_min, v_max), dtype=float)

    pf_base = mean_reverting_series(rng, pf_min, pf_max, count, rho=0.97, step_scale=0.08)
    if multi_unit_pf_min is not None:
        multi_mask = bulbs >= 2
        pf_base = np.where(
            multi_mask,
            np.clip(pf_base, multi_unit_pf_min, pf_max),
            pf_base,
        )

    nominal_w = np.full(count, float(bulb_watts), dtype=float) if unit_watts is None else np.asarray(unit_watts, dtype=float)
    rel_drift = mean_reverting_series(rng, 0.96, 1.03, count, rho=0.985, step_scale=0.05)
    w_unit_base = nominal_w * rel_drift

    v_sub = v_base[:, None] + rng.normal(0.0, 0.12, size=(count, subsamples))
    v_sub = np.round(np.clip(v_sub, 0.0, 260.0), 1)

    pf_sub = pf_base[:, None] + rng.normal(0.0, 0.004, size=(count, subsamples))
    pf_sub = np.round(np.clip(pf_sub, 0.0, 1.0), 2)

    w_sub = w_unit_base[:, None] + rng.normal(0.0, w_unit_base[:, None] * 0.008, size=(count, subsamples))

    # Active units and nominal wattage per sub-sample, with mid-window cuts at transitions
    m_sub = np.repeat(bulbs[:, None], subsamples, axis=1)
    transition_mask = (np.diff(bulbs, prepend=bulbs[0]) != 0) | (
        np.abs(np.diff(nominal_w, prepend=nominal_w[0])) > 1e-6
    )
    for t in np.flatnonzero(transition_mask):
        k_cut = int(rng.integers(1, subsamples))
        m_sub[t, :k_cut] = bulbs[t - 1]
        m_sub[t, k_cut:] = bulbs[t]
        w_sub[t, :k_cut] = w_unit_base[t - 1]
        w_sub[t, k_cut:] = w_unit_base[t]

    is_on_sub = m_sub > 0
    pf_sub = np.where(is_on_sub, pf_sub, 0.0)
    target_p_sub = np.where(is_on_sub, m_sub * w_sub, 0.0)

    if off_current_max > 0.0:
        c_off_base = mean_reverting_series(rng, 0.0, off_current_max, count, rho=0.94, step_scale=0.14)
        c_off_sub = np.round(np.clip(c_off_base[:, None] + rng.normal(0.0, 0.002, size=(count, subsamples)), 0.0, off_current_max), 3)
    else:
        c_off_sub = 0.0

    c_sub = np.where(
        is_on_sub,
        np.round(target_p_sub / np.maximum(v_sub * pf_sub, 1.0), 3),
        c_off_sub,
    )
    p_sub = np.where(is_on_sub, np.round(v_sub * c_sub * pf_sub, 1), 0.0)

    # Firmware averages over valid sub-samples (PZEM004T.ino lines 518-526)
    voltage = np.round(v_sub.mean(axis=1), MEASUREMENT_DECIMALS)
    current = np.round(c_sub.mean(axis=1), MEASUREMENT_DECIMALS)
    power = np.round(p_sub.mean(axis=1), MEASUREMENT_DECIMALS)
    power_factor = np.round(pf_sub.mean(axis=1), MEASUREMENT_DECIMALS)

    # Cumulative energy counter quantized to 0.001 kWh (1 Wh) steps
    energy_raw = energy_start + np.cumsum(power * interval_seconds / 3_600_000.0)
    energy = np.round(np.floor(energy_raw * 1000.0 + 1e-9) / 1000.0, MEASUREMENT_DECIMALS)

    return pd.DataFrame({
        'datetime': timestamps_manila,
        'agent': agent,
        'voltage': voltage,
        'current': current,
        'power': power,
        'energy': energy,
        'power_factor': power_factor,
    })


def generate_data(
    source,
    start,
    duration_hours,
    interval_seconds,
    seed,
    agent,
    break_times=None,
    schedule=None,
    voltage_range=(215.0, 235.0),
    pf_range=(0.60, 0.65),
    bulb_watts=4.75,
    morning_bulbs=(2,),
    afternoon_bulbs=(3, 4),
    jitter_minutes=2.0,
    subsamples=18,
    energy_start=None,
    unit_stages=None,
    off_current_max=0.0,
    multi_unit_pf_min=None,
):
    count = int(round(duration_hours * 3600 / interval_seconds))
    if count < 1:
        raise ValueError('duration_hours must produce at least one sample.')

    rng = np.random.default_rng(seed)

    # Calibrate ON ranges from steady-state source rows if provided, while keeping safe mains envelope
    v_min, v_max = voltage_range
    pf_min, pf_max = pf_range
    if source is not None and not source.empty:
        steady_on = source[(source['power'] > 3.0) & (source['power_factor'] >= 0.45)]
        if not steady_on.empty:
            v_min = min(v_min, float(steady_on['voltage'].quantile(0.01)))
            v_max = max(v_max, float(steady_on['voltage'].quantile(0.99)))
            pf_min = min(pf_min, float(steady_on['power_factor'].quantile(0.01)))
            pf_max = max(pf_max, float(steady_on['power_factor'].quantile(0.99)))
        if energy_start is None:
            energy_start = float(source['energy'].iloc[-1])
    if energy_start is None:
        energy_start = 0.026

    timestamps = pd.date_range(start=start, periods=count, freq=f'{interval_seconds}s')
    if timestamps.tz is None:
        timestamps_manila = timestamps.tz_localize(DATABASE_TIMEZONE)
    else:
        timestamps_manila = timestamps.tz_convert(DATABASE_TIMEZONE)

    if schedule is not None:
        bulbs, unit_watts = build_schedule_bulb_profile(
            timestamps_manila,
            schedule,
            rng,
            morning_bulbs=morning_bulbs,
            afternoon_bulbs=afternoon_bulbs,
            jitter_minutes=jitter_minutes,
            interval_seconds=interval_seconds,
            bulb_watts=bulb_watts,
            unit_stages=unit_stages,
        )
    else:
        default_bulbs = int(rng.choice(afternoon_bulbs))
        bulbs = np.full(count, default_bulbs, dtype=int)
        unit_watts = np.full(count, float(rng.choice(unit_stages)) if unit_stages else float(bulb_watts), dtype=float)

    # Apply explicit --break-times strictly in Asia/Manila
    if break_times:
        local_times = pd.Series(timestamps_manila.time)
        is_break = np.zeros(count, dtype=bool)
        for start_t, end_t in break_times:
            if start_t <= end_t:
                mask = (local_times >= start_t) & (local_times < end_t)
            else:
                mask = (local_times >= start_t) | (local_times < end_t)
            is_break |= mask.values
        bulbs[is_break] = 0

    return synthesize_pzem_window_telemetry(
        timestamps_manila=timestamps_manila,
        bulbs=bulbs,
        rng=rng,
        voltage_range=(v_min, v_max),
        pf_range=(pf_min, pf_max),
        bulb_watts=bulb_watts,
        energy_start=energy_start,
        interval_seconds=interval_seconds,
        subsamples=subsamples,
        agent=agent,
        unit_watts=unit_watts,
        off_current_max=off_current_max,
        multi_unit_pf_min=multi_unit_pf_min,
    )


def write_daily_csvs(data, daily_csv_dir):
    os.makedirs(daily_csv_dir, exist_ok=True)
    local_dates = data['datetime'].dt.tz_convert(DATABASE_TIMEZONE).dt.date
    written = []
    for day_date, group in data.groupby(local_dates, sort=True):
        day_name = DAY_INDEX_TO_NAME[day_date.weekday()]
        out_path = os.path.join(daily_csv_dir, f'{day_name}_dataset.csv')
        csv_data = group.copy()
        csv_data['datetime'] = csv_data['datetime'].map(format_storage_datetime)
        csv_data.to_csv(out_path, index=False, float_format=f'%.{MEASUREMENT_DECIMALS}f')
        written.append((out_path, len(csv_data)))
    return written


def _ensure_sqlite_schema(connection, load=None):
    if load:
        connection.executescript('''
            CREATE TABLE IF NOT EXISTS load_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                datetime TEXT,
                agent TEXT,
                load TEXT,
                voltage REAL,
                current REAL,
                power REAL,
                energy REAL,
                power_factor REAL,
                synced INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_load_readings_agent_load_datetime
                ON load_readings(agent, load, datetime);
        ''')
        view_name = LOAD_VIEWS.get(load)
        if view_name:
            connection.execute(f'''
                CREATE VIEW IF NOT EXISTS {view_name} AS
                SELECT id, datetime, agent, voltage, current, power, energy, power_factor, synced
                FROM load_readings WHERE load = '{load}';
            ''')
    else:
        connection.executescript('''
            CREATE TABLE IF NOT EXISTS energy_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                datetime TEXT,
                agent TEXT,
                voltage REAL,
                current REAL,
                power REAL,
                energy REAL,
                power_factor REAL,
                synced INTEGER DEFAULT 0
            );
        ''')


def write_sqlite(data, database_path, load=None, replace=False):
    connection = sqlite3.connect(database_path)
    _ensure_sqlite_schema(connection, load=load)

    if load:
        if replace:
            connection.execute('DELETE FROM load_readings WHERE load = ?', (load,))
        connection.executemany(
            '''INSERT INTO load_readings
               (datetime, agent, load, voltage, current, power, energy, power_factor, synced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)''',
            [
                (
                    format_storage_datetime(row.datetime),
                    row.agent,
                    load,
                    float(row.voltage),
                    float(row.current),
                    float(row.power),
                    float(row.energy),
                    float(row.power_factor),
                )
                for row in data.itertuples(index=False)
            ],
        )
    else:
        if replace:
            connection.execute('DELETE FROM energy_data')
        connection.executemany(
            '''INSERT INTO energy_data
               (datetime, agent, voltage, current, power, energy, power_factor, synced)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1)''',
            [
                (
                    format_storage_datetime(row.datetime),
                    row.agent,
                    float(row.voltage),
                    float(row.current),
                    float(row.power),
                    float(row.energy),
                    float(row.power_factor),
                )
                for row in data.itertuples(index=False)
            ],
        )
    connection.commit()
    connection.close()


def write_api(data, url, api_key, load=None):
    for row in data.itertuples(index=False):
        body = {
            'datetime': format_storage_datetime(row.datetime),
            'agent': row.agent,
            'voltage': float(row.voltage),
            'current': float(row.current),
            'power': float(row.power),
            'energy': float(row.energy),
            'power_factor': float(row.power_factor),
        }
        if load:
            body['load'] = load
        payload = json.dumps(body).encode('utf-8')
        request = urllib.request.Request(
            url,
            data=payload,
            headers={'Content-Type': 'application/json', 'x-api-key': api_key},
            method='POST',
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f'API rejected synthetic row: HTTP {response.status}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate PZEM-004T firmware-accurate telemetry aligned with Asia/Manila schedules.'
    )
    parser.add_argument('--source', default=None, help='Optional CSV or SQLite database to seed ranges/energy.')
    parser.add_argument('--schedule', default=None, help='Path to schedule markdown (e.g. ../docs/lights_schedule.md).')
    parser.add_argument('--start', required=True, help='Start timestamp (Asia/Manila if naive, e.g. 2026-10-05T00:00:00+08:00).')
    parser.add_argument('--duration-hours', type=float, default=None, help='Duration in hours (default: 168 with --schedule, else 3).')
    parser.add_argument('--interval-seconds', type=int, default=5)
    parser.add_argument('--agent', default='B1-GF-Room1')
    parser.add_argument('--load', choices=['lights', 'fan', 'outlet', 'aircon'], default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--break-times', default='', help='Comma-separated Asia/Manila HH:MM-HH:MM break ranges.')
    parser.add_argument('--bulb-watts', type=float, default=4.75, help='Nominal watts per unit (when --unit-stages omitted).')
    parser.add_argument('--unit-stages', default=None, help='Comma-separated nominal watts per unit speed stage (e.g. 40.3,44.0,48.0 for fans).')
    parser.add_argument('--morning-bulbs', default=None, help='Comma-separated unit count choices for <12:00 Manila.')
    parser.add_argument('--afternoon-bulbs', default=None, help='Comma-separated unit count choices for >=12:00 Manila.')
    parser.add_argument('--voltage-range', default='215.0:235.0', help='Normal mains voltage min:max envelope.')
    parser.add_argument('--pf-range', default=None, help='Normal ON power factor min:max envelope.')
    parser.add_argument('--off-current-max', type=float, default=None, help='Max OFF-state standby CT current noise in Amps (default 0.035 for fan, 0.0 for others).')
    parser.add_argument('--schedule-jitter-minutes', type=float, default=2.0)
    parser.add_argument('--subsamples', type=int, default=18, help='PZEM 250ms sub-samples averaged per 5s window.')
    parser.add_argument('--energy-start', type=float, default=None, help='Starting cumulative energy in kWh.')
    parser.add_argument('--mode', choices=['csv', 'sqlite', 'api'], default='csv')
    parser.add_argument('--output', default='generated_energy_data.csv')
    parser.add_argument('--daily-csv-dir', default=None, help='If set, writes <Day>_dataset.csv files per Manila day.')
    parser.add_argument('--database', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'local.db')))
    parser.add_argument('--replace', action='store_true', help='In sqlite mode, clear existing rows for --load before inserting.')
    parser.add_argument('--api-url', default='http://localhost:5000/api/data')
    parser.add_argument('--api-key', default=os.getenv('HARDWARE_API_KEY'))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.source is None and args.schedule is None:
        raise ValueError('Either --source or --schedule must be provided.')

    duration_hours = args.duration_hours
    if duration_hours is None:
        duration_hours = 168.0 if args.schedule else 3.0

    source = load_source(args.source, agent=args.agent, load=args.load) if args.source else None
    schedule = parse_schedule_markdown(args.schedule) if args.schedule else None

    start = pd.Timestamp(args.start)
    if start.tzinfo is None:
        start = start.tz_localize(DATABASE_TIMEZONE)
    else:
        start = start.tz_convert(DATABASE_TIMEZONE)

    is_fan = args.load == 'fan'
    morning_spec = args.morning_bulbs or ('1,2,3,4' if is_fan else '2')
    afternoon_spec = args.afternoon_bulbs or ('1,2,3,4' if is_fan else '3,4')
    pf_spec = args.pf_range or ('0.94:1.00' if is_fan else '0.60:0.65')
    stage_spec = args.unit_stages or ('40.3,44.0,48.0' if is_fan else None)
    off_current_max = args.off_current_max if args.off_current_max is not None else (0.035 if is_fan else 0.0)
    multi_unit_pf_min = 0.975 if is_fan else None

    break_times = parse_break_times(args.break_times)
    voltage_range = parse_range_pair(args.voltage_range)
    pf_range = parse_range_pair(pf_spec)
    morning_bulbs = parse_bulb_choices(morning_spec)
    afternoon_bulbs = parse_bulb_choices(afternoon_spec)
    unit_stages = parse_stage_watts(stage_spec) if stage_spec else None

    data = generate_data(
        source=source,
        start=start,
        duration_hours=duration_hours,
        interval_seconds=args.interval_seconds,
        seed=args.seed,
        agent=args.agent,
        break_times=break_times,
        schedule=schedule,
        voltage_range=voltage_range,
        pf_range=pf_range,
        bulb_watts=args.bulb_watts,
        morning_bulbs=morning_bulbs,
        afternoon_bulbs=afternoon_bulbs,
        jitter_minutes=args.schedule_jitter_minutes,
        subsamples=args.subsamples,
        energy_start=args.energy_start,
        unit_stages=unit_stages,
        off_current_max=off_current_max,
        multi_unit_pf_min=multi_unit_pf_min,
    )

    if args.daily_csv_dir:
        written = write_daily_csvs(data, args.daily_csv_dir)
        for path, n in written:
            print(f'Wrote {n} rows -> {path}')

    if args.mode == 'csv':
        if not args.daily_csv_dir or args.output != 'generated_energy_data.csv':
            csv_data = data.copy()
            csv_data['datetime'] = csv_data['datetime'].map(format_storage_datetime)
            csv_data.to_csv(args.output, index=False, float_format=f'%.{MEASUREMENT_DECIMALS}f')
            print(f'Wrote {len(csv_data)} rows -> {args.output}')
    elif args.mode == 'sqlite':
        write_sqlite(data, args.database, load=args.load, replace=args.replace)
        target_table = f"load_readings (load={args.load})" if args.load else "energy_data"
        print(f'Inserted {len(data)} rows into {args.database} [{target_table}] (replace={args.replace}).')
    else:
        if not args.api_key:
            raise ValueError('API mode requires --api-key or HARDWARE_API_KEY.')
        write_api(data, args.api_url, args.api_key, load=args.load)

    print(f'Generated {len(data)} normal telemetry rows (final energy={data["energy"].iloc[-1]:.3f} kWh).')
    print(f'Manila Range: {data.datetime.iloc[0].isoformat()} to {data.datetime.iloc[-1].isoformat()}')


if __name__ == '__main__':
    main()
