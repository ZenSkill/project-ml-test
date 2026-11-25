"""Genera clientes_andinatel_raw.csv: 1,250 clientes con ruido intencional en los tipos de datos."""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
n = 1250
fecha_corte = pd.Timestamp('2026-11-01')

planes = ['Basico', 'Intermedio', 'Premium', 'Ultra']
plan = np.random.choice(planes, n, p=[0.35, 0.30, 0.22, 0.13])
cargo_base = pd.Series(plan).map({'Basico': 49.9, 'Intermedio': 79.9, 'Premium': 119.9, 'Ultra': 169.9}).values
cargo = np.round(cargo_base + np.random.normal(0, 12, n), 2)

contrato = np.random.choice(['Mensual', 'Anual_12M', 'Bianual_24M'], n, p=[0.50, 0.30, 0.20])
pago = np.random.choice(['Debito_Automatico', 'App_Bancaria', 'Agente_Efectivo'], n, p=[0.35, 0.40, 0.25])
antiguedad = np.random.randint(30, 2200, n)
fechas = fecha_corte - pd.to_timedelta(antiguedad, unit='D')
consumo = np.round(np.random.gamma(3, 12, n), 1)
reclamos = np.random.poisson(1.2, n)
score = np.round(np.random.normal(620, 90, n)).clip(300, 850)
soporte = np.random.choice([1, 0], n, p=[0.3, 0.7])

# Probabilidad de fuga con correlaciones reales
logit = (-1.40
         + 1.4 * (contrato == 'Mensual') - 0.9 * (contrato == 'Bianual_24M')
         + 0.012 * (cargo - 100)
         + 0.45 * reclamos
         - 0.0007 * antiguedad
         + 0.6 * (pago == 'Agente_Efectivo') - 0.4 * (pago == 'Debito_Automatico')
         - 0.004 * (score - 620)
         - 0.5 * soporte)
fuga = (np.random.rand(n) < 1 / (1 + np.exp(-logit))).astype(int)

# ---- Ruido intencional ----
fecha_txt = [f.strftime('%Y-%m-%d') if np.random.rand() < 0.5 else f.strftime('%d/%m/%Y') for f in fechas]
cargo_txt = [f'S/ {c:.2f}' if np.random.rand() < 0.5 else f'S/{c:.2f}' for c in cargo]
cargo_txt = [np.nan if np.random.rand() < 0.04 else c for c in cargo_txt]
consumo_txt = ['N/D' if np.random.rand() < 0.05 else f'{c} GB' for c in consumo]
reclamos_txt = []
for r in reclamos:
    u = np.random.rand()
    if r == 0 and u < 0.4:
        reclamos_txt.append('ninguno')
    elif u < 0.1:
        reclamos_txt.append(f'{r}.0')
    elif u < 0.13:
        reclamos_txt.append(np.nan)
    else:
        reclamos_txt.append(str(r))
score = score.astype(object)
score[np.random.rand(n) < 0.05] = -999
score[np.random.rand(n) < 0.03] = np.nan
soporte_txt = [np.random.choice(['Si', '1', 'True']) if s == 1 else np.random.choice(['No', '0', 'False']) for s in soporte]
fuga_txt = [np.random.choice(['Si', '1']) if f == 1 else np.random.choice(['No', '0']) for f in fuga]

df = pd.DataFrame({
    'ID_Cliente': [f'CLI-{i:05d}' for i in range(1, n + 1)],
    'Fecha_Alta': fecha_txt,
    'Plan_Tarifario': plan,
    'Cargo_Mensual': cargo_txt,
    'Consumo_Datos_GB': consumo_txt,
    'Reclamos_6M': reclamos_txt,
    'Metodo_Pago': pago,
    'Tipo_Contrato': contrato,
    'Score_Crediticio': score,
    'Soporte_Prioritario': soporte_txt,
    'Fuga_Cliente': fuga_txt,
})
ruta = Path(__file__).resolve().parent / 'clientes_andinatel_raw.csv'
df.to_csv(ruta, index=False)
print(f'Archivo generado: {ruta} ({len(df)} registros)')
print('Tasa de fuga: {:.1%}'.format(fuga.mean()))
