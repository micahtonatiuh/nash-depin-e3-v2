# R1.4 — Prueba física con cinta aleatoria (revisión Paper 1, Access-2026-41887)

Archivos **nuevos**. No se modifica `experiments/run_e5.py` ni `nash_depin/*`.

| Archivo | Qué hace |
|---|---|
| `tape_core.py` | Cinta aleatoria, regla de decisión y referencia de simulación (importa la lógica de `run_e5.py`) |
| `run_e5_tape.py` | Corrida física con la cinta; lazo cerrado opcional; verificación manual de celda |
| `test_tape.py` | Prueba sin robots: reproduce paso a paso las trayectorias físicas del E3 (semillas 2 y 4) |
| `check_refs.py` | Compara las referencias generadas en el lab con las de la nube (`expected/`) |
| `qtables/` | Q-tables E2 de las semillas 0 y 4 (SHA-256 idénticos a `nash-depin-e2-v2/qtables_all_seeds`) |
| `expected/seed0`, `expected/seed4` | Referencias de la nube, 10 episodios por semilla |

## Por qué esta prueba

En `run_e5.py` el estado avanza con las celdas **comandadas**, así que la "fidelidad lógica" del E3 es 100% por construcción. Aquí:

- La política es **estocástica** (muestreo de σ + ε = 0.05), con lo que aparecen traspasos de la tarea. Toda la aleatoriedad sale de una cinta fija, así que la referencia de simulación es única y se calcula **antes** de mover los robots.
- Con `--loop closed` la siguiente decisión usa las celdas **medidas** por odometría. Si un robot no llega a su celda, la política ve otro estado y la trayectoria se separa de la referencia. La fidelidad deja de ser automática.
- Con `--manual_check` el operador anota al final la celda real de cada robot (verdad externa, no odometría).

Inicio: robot 0 en (0,0) y robot 1 en (4,4), el mismo inicio de entrenamiento y evaluación del E2 (Tabla 1).

## 0. Llevar los archivos a la PC del lab

En Windows, dentro de `RevisionBCOct26\nash-depin-e5`, agrega **solo** esta carpeta (el resto aparece "modificado" únicamente por los finales de línea CRLF de Windows; no lo agregues):

```bash
git add revision_r14
git commit -m "R1.4: tape-driven physical test (new files only)"
git push
```

En la PC del lab: `cd ~/nash-depin-e5 && git pull`.

## 1. Verificación sin robots (en la PC del lab, ~3 min)

```bash
cd ~/nash-depin-e5
python3 revision_r14/test_tape.py                        # debe decir ALL PASS
for s in 0 4; do
  python3 revision_r14/run_e5_tape.py --config config/e5_config_tb4x2.yaml \
    --seed $s --qdir revision_r14/qtables --starts 0 0 4 4 --episodes 10 \
    --output /tmp/r14_check/seed$s --sim_only
  python3 revision_r14/check_refs.py /tmp/r14_check/seed$s/reference revision_r14/expected/seed$s
done                                                     # ALL PASS en ambas
```

Si algo dice FAIL, **no muevas los robots** y mándame la salida.

## 2. Arena y regresión con el código original (~10 min)

1. Arma y calibra la arena (celda de 0.618 m) como en el E3.
2. Regresión determinista con el script original, sin cambios:
   ```bash
   python3 experiments/run_e5.py --config config/e5_config_tb4x2.yaml \
     --output r14_results/regress_seed2 --seed 2 --phase frozen --dry_run
   python3 experiments/compare_traj.py sim_traj_seed2.json r14_results/regress_seed2/traj_seed2_ep1.json
   ```
   Esperado: 195 tareas para R0, 200/200 pasos idénticos y error terminal < 0.15 m.

## 3. Corrida física (~3–4 min por episodio)

Usa la **misma rutina de odometría entre episodios que en el E3** (el script asume odometría en cero en la celda de inicio de cada robot, igual que `run_e5.py`). Al inicio de cada episodio imprime las celdas medidas; si no coinciden con las esperadas, se detiene y pide confirmación.

Robot 0 (`tb1`) en (0,0), sobre la tarea. Robot 1 (`tb2`) en (4,4), en la esquina opuesta.

```bash
# Semilla 4 (la misma Q-table del E3 físico). 10 episodios, el 1.º a velocidad de seguridad.
python3 revision_r14/run_e5_tape.py --config config/e5_config_tb4x2.yaml \
  --seed 4 --qdir revision_r14/qtables --starts 0 0 4 4 --episodes 10 \
  --loop closed --manual_check --output r14_results/seed4

# Semilla 0 (la que más traspasos tiene: 24 en 10 episodios).
python3 revision_r14/run_e5_tape.py --config config/e5_config_tb4x2.yaml \
  --seed 0 --qdir revision_r14/qtables --starts 0 0 4 4 --episodes 10 \
  --loop closed --manual_check --output r14_results/seed0
```

- Batería baja: `Ctrl+C`, cambia la batería y repite el mismo comando con `--resume`.
- Al final de cada episodio escribe la celda real de cada robot (`fila col`), viéndola en la cinta del piso.
- Si un episodio se detiene por "celdas medidas inválidas", recoloca los robots y continúa; queda registrado.
- **Si el tiempo no alcanza:** prioriza la semilla 4 completa (10 episodios), luego la semilla 0 con los que se puedan.

## 4. Qué traer de vuelta

La carpeta `r14_results/` completa (`episode_metrics.csv`, `phy_ep*.json`, `reference/`, `manifest.json`, `experiment.log`) y la salida de los pasos 1 y 2.

## Referencias esperadas (simulación, inicio (0,0)/(4,4), ε = 0.05)

| Semilla | Episodios con traspaso | Traspasos totales | Movimientos por episodio | Gini por episodio (mediana) |
|---|---|---|---|---|
| 4 | 9/10 | 9 | 20 | 0.15 |
| 0 | 8/10 | 24 | 27 | 0.16 |

Las métricas que reporta el CSV por episodio: `logical_match` (trayectoria idéntica a la referencia), `first_divergence`, `cell_track_rate` (fracción de pasos en que la celda medida = comandada), error mediano y máximo por paso, error terminal, `manual_ok` y tareas/traspasos/Gini.
