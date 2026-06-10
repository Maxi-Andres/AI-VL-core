# VLM PoC — Inspección industrial con Ollama

Detección de instrumentos/objetos en imágenes usando un VLM (qwen3-vl) servido
por Ollama, devolviendo JSON con bounding boxes. Pensado para validar el VLM
primario del PoC (criterios F1.8) y el contrato VLM→VLA.

## Estructura del proyecto

```
.
├── menu.py            # entrada principal (menú interactivo)
├── src/               # código fuente
│   ├── vlm_common.py  # núcleo: prompts, cliente Ollama, parseo, config
│   ├── smoke_test.py  # smoke test de 1 imagen (CLI)
│   └── benchmark.py   # benchmark de latencia/JSON + A/B de prompts (CLI)
├── fotos/             # carpetas de imágenes
│   ├── clean/         # set de prueba del lab
│   └── ciudad/        # set adicional
├── results/           # salida de los benchmarks (JSON)
├── config.json        # configuración persistente (se crea sola)
└── requirements.txt
```

## Requisitos

```bash
pip install -r requirements.txt    # (solo requests)
```

- **Ollama** corriendo en `http://localhost:11434` con los modelos descargados
  (`ollama pull qwen3-vl:4b`, etc.).
- Verificá que esté vivo: `curl http://localhost:11434/api/version`
- Verificá que el modelo cargue **100% en GPU**: `ollama ps` (ver nota de VRAM abajo).

## Uso rápido: el menú (sin escribir comandos)

```bash
python3 menu.py        # o simplemente darle Play al archivo en el IDE
```

El menú tiene dos formas de analizar:

1. **Smoke test** (opción 1): una imagen, imprime el razonamiento en vivo + JSON.
2. **Benchmark** (opción 2): abre un **submenú** donde elegís **qué imágenes**
   (cuáles y cuántas), **qué modelos**, **qué prompts** (una o varias variantes),
   cuántas **runs**, y **uno o varios** valores de `max_tokens`, `num_ctx` y `think`.
   Corre el **producto cartesiano** de todas esas dimensiones con **barra de
   progreso** y al final reporta **tiempos** (por imagen, total, promedio,
   P50/P95), la **tasa de JSON válido** y un veredicto
   con la mejor combinación. Comparar varias prompts acá reemplaza al viejo
   `prompt_test`: las prompts se prueban igual que los modelos, todo junto.

**Todo lo que elegís se guarda en `config.json`**, así la próxima vez arranca
con lo último que usaste. El benchmark tiene su **propia** config (claves
`benchmark_*`), independiente del smoke test: podés correr el benchmark con un
contexto más liviano (más rápido) sin bajarle el contexto al smoke test.

---

## Uso sin menú (línea de comandos)

Los dos scripts toman sus **defaults de `config.json`**; cualquier flag pisa ese
default solo para esa corrida (no modifica el archivo).

### Smoke test — una imagen

```bash
python3 src/smoke_test.py                          # usa todo lo de config.json
python3 src/smoke_test.py fotos/clean/5.jpeg        # otra imagen
python3 src/smoke_test.py fotos/clean/5.jpeg --model qwen3-vl:8b
python3 src/smoke_test.py fotos/clean/5.jpeg --scope todo
python3 src/smoke_test.py fotos/clean/5.jpeg --max-tokens 8192 --num-ctx 16384
python3 src/smoke_test.py fotos/clean/5.jpeg --no-think     # pedir think=false
```

El smoke test **imprime en vivo lo que el modelo va pensando** (en gris) y al
final el JSON, la latencia y los tokens de entrada/salida.

| Flag            | Default (config.json) | Qué hace |
|-----------------|-----------------------|----------|
| `image` (posic.)| `image`               | Ruta a la imagen. Si la omitís, usa la de config. |
| `--model`       | `model`               | Modelo de Ollama (ej. `qwen3-vl:4b`, `qwen3-vl:8b`, `qwen2.5vl:7b`). |
| `--scope`       | `scope`               | `industrial` (instrumentos industriales) o `todo` (cualquier objeto). |
| `--variant`     | `variant`             | Variante de prompt (`v1_original`, `v2_antiloop`, …). Ver "Variantes de prompt". |
| `--max-tokens`  | `max_tokens`          | Tope de tokens de **salida** (`num_predict`; incluye el razonamiento). |
| `--num-ctx`     | `num_ctx`             | Ventana de contexto (entrada+salida); la que muestra `ollama ps`. |
| `--think` / `--no-think` | `think`      | Pide razonar/no. **Solo aplica a modelos con capability `thinking`** y solo se manda si el modelo la tiene. En `qwen3-vl` (0.30.6) `--no-think` se ignora (razona igual); para no razonar, usá `qwen2.5vl:7b`. |
| `--url`         | `url`                 | Host de Ollama. |

### Benchmark — conjunto de imágenes (tiempos + P50/P95 + % JSON válido)

El benchmark **barre el producto cartesiano de TODAS estas dimensiones** —
modelos, prompts, imágenes, `max_tokens`, `num_ctx` y `think` — y cada
combinación es una fila del reporte. Todas aceptan **varios valores**:

```bash
python3 src/benchmark.py                                    # usa todo lo de config.json (claves benchmark_*)
python3 src/benchmark.py fotos/clean --runs 5
python3 src/benchmark.py fotos/clean --models qwen3-vl:4b qwen3-vl:8b      # varios modelos
python3 src/benchmark.py fotos/clean --variants v1_original v2_antiloop    # A/B de prompts
python3 src/benchmark.py fotos/clean --max-tokens 4096 8192                # comparar topes de salida
python3 src/benchmark.py fotos/clean --num-ctx 8192 16384                  # comparar ventanas de contexto
python3 src/benchmark.py fotos/clean --think true false                    # comparar con/sin razonar
python3 src/benchmark.py fotos/clean --images 1.jpeg 14.jpeg 16.jpeg       # solo esas imágenes
python3 src/benchmark.py fotos/clean --scope todo --runs 1
```

> Cuidado con la **explosión combinatoria**: el total de llamadas es
> `imágenes × runs × modelos × prompts × max_tokens × num_ctx × think`. El menú te
> muestra el total antes de correr.

| Flag            | Default (config.json)  | Qué hace |
|-----------------|------------------------|----------|
| `folder` (posic.)| `folder`              | Carpeta con imágenes (jpg/jpeg/png/bmp/webp). |
| `--images`      | (todas)                | Nombres concretos dentro de la carpeta (ej. `1.jpeg 14.jpeg`). Sin esto, usa todas. |
| `--models`      | `benchmark_models`     | Uno o varios modelos a comparar (separados por espacio). |
| `--variants`    | `benchmark_variants`   | Una o varias variantes de prompt (`v1_original v2_antiloop …`). |
| `--runs`        | `benchmark_runs`       | Repeticiones por imagen. |
| `--scope`       | `benchmark_scope`      | `industrial` o `todo`. |
| `--max-tokens`  | `benchmark_max_tokens` | Uno o **varios** topes de salida (`num_predict`) a comparar (ej. `4096 8192`). |
| `--num-ctx`     | `benchmark_num_ctx`    | Una o **varias** ventanas de contexto a comparar (ej. `8192 16384`). |
| `--think`       | `benchmark_think`      | Uno o **varios** valores `true`/`false` a comparar (ej. `--think true false`). Solo cambia algo en modelos con capability `thinking`; ver nota de razonamiento. |
| `--url`         | `url`                  | Host de Ollama. |

Mientras corre muestra una **barra de progreso** (combinación/imagen actual, %
y ETA). Al terminar imprime el **tiempo por imagen** (prom/min/max) y una tabla con
**una fila por combinación** y columnas `ctx`/`maxtok`/`thk` además de
P50/P95/media/min/max/total + JSON% + cortes por `length` + objetos promedio; cierra
con un **veredicto** que te da la **config completa** de la mejor combinación
(model + variant + max_tokens + num_ctx + think), listo para pegar en `config.json`.
Guarda todo en `results/benchmark_resultados.json`.

> **Nota de contexto (velocidad):** el benchmark arranca con `num_ctx=8192` /
> `max_tokens=4096` — la **mitad** de lo que usa el smoke test (16384 / 8192).
> Menos contexto = prefill más rápido (la imagen entra a menor resolución). Con
> el prompt `v2_antiloop` esto **no** trunca: probado en las imágenes 1, 14 y 16
> (incluida la que antes se quedaba sin contexto) → JSON válido y ~13–17 s.
> Si una imagen difícil se vacía (`finish_reason: length`), subí `num_ctx` /
> `max_tokens` desde el submenú (opción 6).

> **Comparar prompts (A/B):** el A/B de variantes de prompt **ya no es un script
> aparte** — está dentro del benchmark. Pasale varias con `--variants` (o elegilas
> en el submenú, opción 5) y compará velocidad / JSON% / objetos dejando todo lo
> demás constante. Ej.: `python3 src/benchmark.py fotos/clean --variants v1_original v2_antiloop`.

---

## Modos de detección (`scope`)

| scope        | Qué detecta | Taxonomía (`tipo`) |
|--------------|-------------|--------------------|
| `industrial` | **Cualquier** instrumento/equipo industrial | familia general: `presion\|temperatura\|caudal\|nivel\|electrica\|analisis\|control\|vibracion\|valvula\|epp\|otro` (+ `descripcion` libre con el detalle) |
| `todo`       | Cualquier objeto visible | categoría libre (`persona`, `vehiculo`, …) |

Los prompts de cada modo están en `src/vlm_common.py` → `SCOPES`. El modo
`industrial` da una lista de instrumentos típicos por familia (manómetro,
termopar, caudalímetro, sensor radar, etc.) **como referencia, no como lista
cerrada**: el modelo debe poder reconocer cualquier instrumento de industria.

Los **bounding boxes** se devuelven normalizados 0–1. qwen3-vl los entrega en
píxeles del archivo original, así que el código los normaliza solo (leyendo el
tamaño real de la imagen del header JPEG/PNG).

## Variantes de prompt (intercambiables / A-B test)

Los prompts viven en `src/vlm_common.py` → `PROMPT_VARIANTS`, **uno por variante**, y
están escritos en **inglés** (qwen3-vl razona en inglés; menos overhead de
traducción). Las *keys* del JSON y los valores de `tipo` siguen en español porque
son el contrato VLM→VLA.

| scope | variante | cómo es |
|-------|----------|---------|
| `industrial` | `v1_original` | El prompt corto original. Da la lista de familias y dice "no dudes en la categoría". |
| `industrial` | `v2_antiloop` | Más larga y explícita: aclara que **los equipos cuentan** (no solo instrumentos), amplía `electrica` (transformador, bushing, seccionador…) y arranca con una REGLA anti-deliberación para que **no se trabe eligiendo categoría** (era lo que vaciaba el `content`). |
| `todo` | `default` | Único prompt para objetos genéricos. |

**Cómo intercambiarlas** (3 formas, no hace falta tocar código salvo la última):
1. `config.json` → clave `"variant"` (smoke) o `"benchmark_variants"` (benchmark).
2. Flag `--variant v2_antiloop` en `src/smoke_test.py`, o `--variants v1_original v2_antiloop` en `src/benchmark.py` (pisa la config para esa corrida).
3. La variante **activa por defecto** está en `DEFAULT_VARIANT` (`src/vlm_common.py`); cambiala ahí si querés mover el default global.

Para **agregar** una variante nueva: sumá una entrada a `PROMPT_VARIANTS["industrial"]`
y compará con `python3 src/benchmark.py … --variants <vieja> <nueva>`.

> **Nota de medición:** en pruebas sobre las imágenes 1, 14 y 16, `v2_antiloop`
> resultó **más rápida** que `v1_original` (p. ej. en la 16: ~15 s vs ~97 s),
> porque cortar la deliberación de categoría ahorra muchos tokens de razonamiento.
> La variante activa por defecto es `v1_original` (pedido explícito); cambiala a
> `v2_antiloop` si querés la más rápida. Reproducí con
> `python3 src/benchmark.py fotos/clean --variants v1_original v2_antiloop`.

## config.json

Se crea solo la primera vez. Lo edita el menú, pero podés tocarlo a mano:

```json
{
  "model": "qwen3-vl:4b",
  "image": "fotos/clean/2.jpeg",
  "folder": "fotos/clean",
  "scope": "industrial",
  "variant": "v1_original",
  "max_tokens": 8192,
  "num_ctx": 16384,
  "think": true,
  "url": "http://localhost:11434",

  "benchmark_models": ["qwen3-vl:8b", "qwen3-vl:4b", "qwen2.5vl:7b"],
  "benchmark_runs": 3,
  "benchmark_images": [],
  "benchmark_scope": "industrial",
  "benchmark_variants": ["v2_antiloop"],
  "benchmark_max_tokens": [4096],
  "benchmark_num_ctx": [8192],
  "benchmark_think": [true]
}
```

Las dimensiones de barrido del benchmark (`benchmark_models`, `benchmark_variants`,
`benchmark_max_tokens`, `benchmark_num_ctx`, `benchmark_think`) son **listas**:
poné un solo elemento para no barrer esa dimensión, o varios para compararlos
(ej. `"benchmark_max_tokens": [4096, 8192]`). El benchmark también acepta valores
escalares de configs viejas (los envuelve en lista solo).

Las claves `benchmark_*` son la config **propia del benchmark** (independiente
del smoke test). `benchmark_images: []` significa **todas** las imágenes de la
carpeta (si agregás fotos, entran solas); poné una lista de nombres
(`["1.jpeg", "16.jpeg"]`) para correr solo esas. El smoke test (`max_tokens`,
`num_ctx`, `scope`, `variant`) queda intacto.

## Notas importantes

- **VRAM / elección de modelo:** en una GPU de 8 GB (ej. RTX 5060), `qwen3-vl:8b`
  (~10 GB cargado) **no entra** y Ollama lo parte ~53% CPU / 47% GPU →
  ~85–110 s por imagen. `qwen3-vl:4b` (~3.3 GB) carga **100% en GPU** →
  ~15–25 s. Por eso el default es `4b`. Confirmá con `ollama ps`.
- **Razonamiento (thinking) — el flag NO es un on/off, lo decide el MODELO:**
  probado contra Ollama 0.30.6, el control de razonamiento **no** es un simple
  `think: true/false` por request, sino que depende de la *capability* del modelo
  (la podés ver con `ollama show <modelo>` o `/api/show`):

  | modelo | capability `thinking` | `--think` (ON) | `--no-think` (OFF) |
  |--------|:---------------------:|----------------|--------------------|
  | `qwen3-vl:4b` / `:8b` | sí | razona | **razona igual** (el renderer `qwen3-vl-thinking` **ignora** `think:false` en 0.30.6) |
  | `qwen2.5vl:7b` | no | **error HTTP 400** si se manda el flag | nunca razona (OFF real) |

  O sea: **el verdadero interruptor de razonamiento es elegir el modelo**.
  `qwen3-vl` siempre piensa; si querés que **no** piense, usá un modelo sin la
  capability `thinking` (ej. `qwen2.5vl:7b`). Por eso el código consulta
  `/api/show` y **sólo manda `think` si el modelo lo soporta** (a `qwen2.5vl` no se
  lo manda: si no, devuelve `400 "does not support thinking"` — antes eso hacía
  fallar el benchmark de ese modelo). La UI te dice la verdad por modelo (si
  razona, si se pidió y si efectivamente razonó). Si necesitás apagar el
  razonamiento de `qwen3-vl`, hace falta otra versión de Ollama (en 0.30.6 no se
  puede). Y ojo: cuando razona, el riesgo es quedarse sin tokens — si el
  razonamiento se come el presupuesto, `content` vuelve vacío
  (`finish_reason: length`); se mitiga con el prompt anti-loop (abajo) y dándole
  aire con `max_tokens` y `num_ctx`. Usamos el endpoint **nativo** (`/api/chat`,
  no `/v1/...`) porque separa el razonamiento (`thinking`) del JSON (`content`) y
  permite **imprimirlo en vivo**.
- **Prompt anti-loop (por qué se quedaba sin contexto):** el caso típico era una
  imagen donde el modelo *reconocía* el objeto (p. ej. un bushing / transformador)
  pero entraba en bucle **debatiendo en qué familia ponerlo** (`electrica`? `otro`?
  `control`?) hasta agotar los tokens → `content` vacío. El prompt de `industrial`
  ahora corta eso de raíz:
  - el `system` le pide **razonar en pocos pasos, sin repetirse, y pasar al JSON
    apenas reconoce el objeto** (no re-evaluar la categoría);
  - el `user` arranca con una **REGLA explícita**: identificar de un vistazo, no
    debatir, y si duda entre dos familias elegir una y poner el detalle en
    `descripcion` (o usar `otro`);
  - se aclara que **los equipos también cuentan** (no solo instrumentos de medición)
    y se ampliaron las familias (`electrica` ahora incluye transformador, bushing,
    seccionador, interruptor, barra, celda; se sumaron ejemplos de `valvula` y `epp`).

  Resultado esperado: razonamiento más corto (más rápido) y sin truncarse. Igual
  conviene dejar margen de tokens para las imágenes difíciles.
- **`max_tokens` vs `num_ctx` (la diferencia que importa):**
  - **`num_ctx`** = la **ventana de contexto completa**: todo lo que entra +
    todo lo que sale. Es decir `entrada (system + user + tokens de la imagen) +
    salida (razonamiento + respuesta)`. Es el número que ves en `ollama ps` bajo
    *context*. Además, **más `num_ctx` deja que Ollama mande la imagen a mayor
    resolución** (más tokens de imagen → más detalle).
  - **`max_tokens`** (= `num_predict`) = el **tope de lo que el modelo *genera***
    (razonamiento + respuesta). Cuando se llega a este tope, corta y devuelve
    `finish_reason: length` (lo que te pasaba: cortaba en pleno razonamiento).
  - **Cómo se combinan:** el presupuesto real de salida es
    `min(max_tokens, num_ctx − tokens_de_entrada)`. O sea, **los dos tienen que
    alcanzar**: si `max_tokens` es chico, corta aunque sobre `num_ctx`; si
    `num_ctx` es chico, la entrada (imagen incluida) le come lugar a la salida y
    también corta. La entrada acá ronda ~1000–2600 tokens, así que con
    `num_ctx 16384` y `max_tokens 8192` quedan holgados los dos.
  - **Defaults actuales:** `max_tokens 8192`, `num_ctx 16384` (antes 4096 / 8192,
    que se quedaban cortos en imágenes difíciles). Subilos más con `--max-tokens`
    / `--num-ctx` si una imagen sigue truncándose; bajalos si querés más velocidad
    y tus imágenes son simples.
- **Latencia vs objetivo F1.8:** el target de **P95 < 1.5 s** no es alcanzable
  con un VLM de esta clase en este hardware (mejor caso ~15 s). Para acercarse
  haría falta otro modelo/cuantización, bajar resolución de imagen, o más VRAM.

## Estructura

| Ruta                  | Qué es |
|-----------------------|--------|
| `menu.py`             | Menú interactivo (entrada principal). |
| `src/smoke_test.py`   | Smoke test de 1 imagen (CLI). |
| `src/benchmark.py`    | Benchmark de latencia/JSON + A/B de prompts (CLI). |
| `src/vlm_common.py`   | Núcleo compartido: prompts (`PROMPT_VARIANTS`/`SCOPES`), cliente Ollama, config. |
| `config.json`         | Configuración persistente (se crea sola, en la raíz). |
| `fotos/clean/`, `fotos/ciudad/` | Imágenes de prueba. |
| `results/`            | Salida de los benchmarks (JSON). |
