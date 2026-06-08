# VLM PoC — Inspección industrial con Ollama

Detección de instrumentos/objetos en imágenes usando un VLM (qwen3-vl) servido
por Ollama, devolviendo JSON con bounding boxes. Pensado para validar el VLM
primario del PoC (criterios F1.8) y el contrato VLM→VLA.

## Requisitos

```bash
pip install requests
```

- **Ollama** corriendo en `http://localhost:11434` con los modelos descargados
  (`ollama pull qwen3-vl:4b`, etc.).
- Verificá que esté vivo: `curl http://localhost:11434/api/version`
- Verificá que el modelo cargue **100% en GPU**: `ollama ps` (ver nota de VRAM abajo).

## Uso rápido: el menú (sin escribir comandos)

```bash
python3 menu.py        # o simplemente darle Play al archivo en el IDE
```

El menú deja elegir modelo, imagen, modo de detección, etc. **Todo lo que
elegís se guarda en `config.json`**, así la próxima vez arranca con lo último
que usaste. No hace falta tocar nada para correrlo de nuevo.

---

## Uso sin menú (línea de comandos)

Los dos scripts toman sus **defaults de `config.json`**; cualquier flag pisa ese
default solo para esa corrida (no modifica el archivo).

### Smoke test — una imagen

```bash
python3 03_smoke_test.py                          # usa todo lo de config.json
python3 03_smoke_test.py fotosClean/5.jpeg        # otra imagen
python3 03_smoke_test.py fotosClean/5.jpeg --model qwen3-vl:8b
python3 03_smoke_test.py fotosClean/5.jpeg --scope todo
python3 03_smoke_test.py fotosClean/5.jpeg --think --max-tokens 8192
```

| Flag            | Default (config.json) | Qué hace |
|-----------------|-----------------------|----------|
| `image` (posic.)| `image`               | Ruta a la imagen. Si la omitís, usa la de config. |
| `--model`       | `model`               | Modelo de Ollama (ej. `qwen3-vl:4b`, `qwen3-vl:8b`, `qwen2.5vl:7b`). |
| `--scope`       | `scope`               | `industrial` (solo instrumentos) o `todo` (cualquier objeto). |
| `--max-tokens`  | `max_tokens`          | Tope de tokens de salida (incluye razonamiento). |
| `--think`       | `think`               | Activa el razonamiento del modelo (más lento). |
| `--url`         | `url`                 | Endpoint de Ollama. |

### Benchmark — carpeta de imágenes (P50/P95, % JSON válido)

```bash
python3 04_benchmark.py                            # usa todo lo de config.json
python3 04_benchmark.py fotosClean --runs 5
python3 04_benchmark.py fotosClean --models qwen3-vl:4b qwen3-vl:8b
python3 04_benchmark.py fotosClean --scope todo --runs 1
```

| Flag            | Default (config.json) | Qué hace |
|-----------------|-----------------------|----------|
| `folder` (posic.)| `folder`             | Carpeta con imágenes (jpg/jpeg/png/bmp/webp). |
| `--models`      | `benchmark_models`    | Lista de modelos a comparar (separados por espacio). |
| `--runs`        | `benchmark_runs`      | Repeticiones por imagen. |
| `--scope`       | `scope`               | `industrial` o `todo`. |
| `--max-tokens`  | `max_tokens`          | Tope de tokens de salida. |
| `--think`       | `think`               | Activa el razonamiento del modelo. |
| `--url`         | `url`                 | Endpoint de Ollama. |

Escribe la tabla comparativa por pantalla y guarda `benchmark_resultados.json`.

---

## Modos de detección (`scope`)

| scope        | Qué detecta | Taxonomía (`tipo`) |
|--------------|-------------|--------------------|
| `industrial` | Solo instrumentos/equipos industriales | `manometro\|termometro\|valvula\|sensor\|epp\|otro` |
| `todo`       | Cualquier objeto visible | categoría libre (`persona`, `vehiculo`, …) |

Los prompts de cada modo están en `vlm_common.py` → `SCOPES`. El modo
`industrial` es más estable; `todo` es más libre y puede variar más entre corridas.

## config.json

Se crea solo la primera vez. Lo edita el menú, pero podés tocarlo a mano:

```json
{
  "model": "qwen3-vl:4b",
  "image": "fotosClean/2.jpeg",
  "folder": "fotosClean",
  "scope": "industrial",
  "max_tokens": 4096,
  "think": false,
  "url": "http://localhost:11434/v1/chat/completions",
  "benchmark_models": ["qwen3-vl:8b", "qwen3-vl:4b", "qwen2.5vl:7b"],
  "benchmark_runs": 3
}
```

## Notas importantes

- **VRAM / elección de modelo:** en una GPU de 8 GB (ej. RTX 5060), `qwen3-vl:8b`
  (~10 GB cargado) **no entra** y Ollama lo parte ~53% CPU / 47% GPU →
  ~85–110 s por imagen. `qwen3-vl:4b` (~3.3 GB) carga **100% en GPU** →
  ~15–25 s. Por eso el default es `4b`. Confirmá con `ollama ps`.
- **Razonamiento (thinking):** `qwen3-vl` razona por defecto y eso gasta
  latencia/tokens (a veces se corta y devuelve `content` vacío). Lo apagamos con
  `"think": false` en el payload (el switch real). Solo activalo con `--think` /
  desde el menú si querés ver el razonamiento.
- **Latencia vs objetivo F1.8:** el target de **P95 < 1.5 s** no es alcanzable
  con un VLM de esta clase en este hardware (mejor caso ~15 s). Para acercarse
  haría falta otro modelo/cuantización, bajar resolución de imagen, o más VRAM.

## Estructura

| Archivo            | Qué es |
|--------------------|--------|
| `menu.py`          | Menú interactivo (entrada principal). |
| `03_smoke_test.py` | Smoke test de 1 imagen (CLI). |
| `04_benchmark.py`  | Benchmark de latencia/JSON (CLI). |
| `vlm_common.py`    | Núcleo compartido: prompts (`SCOPES`), cliente Ollama, config. |
| `config.json`      | Configuración persistente (se crea sola). |
| `fotosClean/`      | Imágenes de prueba. |
