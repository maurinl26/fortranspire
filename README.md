<div align="center">

# 🚀 Fortran → GPU + JAX

**Agent de transformation Fortran scientifique vers GPU (OpenACC) et JAX.**  
Propulsé par Mistral-Large (endpoint souverain) · LangGraph · Loki (ECMWF)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-Ready-green.svg)](https://modelcontextprotocol.io/)
[![Documentation Status](https://readthedocs.org/projects/fortranspire/badge/?version=latest)](https://fortranspire.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.0000000.svg)](https://doi.org/10.5281/zenodo.0000000)
[![JOSS draft](https://github.com/maurinl26/fortranspire/actions/workflows/draft-paper.yml/badge.svg)](https://github.com/maurinl26/fortranspire/actions/workflows/draft-paper.yml)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

</div>

---

## 0. 🎯 Le besoin R&D scientifique

### Le contexte

Les équipes R&D de TotalEnergies font tourner des **simulations physiques lourdes** (sismique,
géomécanique, réservoir, CFD) pour trois usages : exploration, optimisation de production,
validation des modèles IA. Ces simulations reposent sur des codes Fortran écrits dans les années
90–2000 qui tournent sur des **clusters CPU HPC** (Pangea) en mode batch — parfois plusieurs
jours par run.

> **Expertise transférable — Météo France**  
> Ce problème n'est pas propre à l'E&P. Les codes de prévision numérique du temps (NWP) — ARPEGE,
> AROME, IFS (ECMWF) — partagent exactement le même patrimoine : Fortran 90 monolithique, COMMON
> blocks, clusters CPU massivement parallèles (MPI), et une dette technique qui freine
> l'intégration GPU et l'IA. La même chaîne de valeur s'applique : portage GPU → génération de
> données → surrogate IA → prévision hybride physique/ML. Les patterns de transformation
> documentés ici (INTENT, SAVE, COMMON, OpenACC) sont directement applicables aux codes
> météorologiques.  
> — **Loïc Maurin** · [LinkedIn](https://www.linkedin.com/in/lo%C3%AFc-maurin/) · maurin.loic.ac@gmail.com

### Les pain points

**1 — Coût humain du portage GPU**  
Le patrimoine logiciel R&D compte des centaines de codes Fortran legacy. Les porter manuellement
sur GPU demande 2 à 6 semaines par code et un profil rare (HPC senior + OpenACC + Cython).
Résultat : les GPU restent sous-utilisés, les équipes gardent des workflows CPU lents.

**2 — Goulot d'étranglement sur la génération de données IA**  
Entraîner un surrogate IA (FNO, PINN) nécessite des dizaines de milliers de runs de simulation.
Sur CPU Pangea, un jeu d'entraînement prend des semaines. Sur GPU, quelques heures.
Le portage GPU est le **bloquant principal** de la boucle Simulation → IA.

**3 — Validation physique des modèles IA coûteuse**  
Les modèles IA (proxy de décision) doivent être vérifiés par le code physique de référence à
chaque itération d'entraînement (loss physique, métriques de conservation). Si ce code est lent,
le cycle de validation bride la cadence d'entraînement.

**4 — Rigidité des codes legacy**  
Les codes Fortran monolithiques ne s'interfacent pas avec Python, JAX, ou les pipelines MLOps
modernes. Pas d'API, pas de bindings — les scientifiques ne peuvent pas les appeler depuis un
notebook ou un pipeline Airflow.

### La chaîne de valeur

```
Code Fortran Legacy (HPC Pangea, CPU multi-cœur, jours/run)
    │
    ▼  [Phase 1 — cet agent, ~2 min]
Code GPU cloud (A100/T4, ×10–×100 speedup, heures/run)
    │  ├─ Génère les données d'entraînement du surrogate IA
    │  └─ Valide les outputs critiques des modèles IA (loss physique)
    │
    ▼  [Phase 2 — différentiation automatique JAX]
Surrogate IA (FNO, PINN — speedup ×10⁴–×10⁵ vs simulation FD)
    │  ├─ Proxy décisionnel instantané (exploration, optimisation)
    │  └─ Modèle de ciblage pour orienter le simulateur numérique
    │
    ▼  [Pipeline MLOps]
Modèle de décision (inversion sismique, optimisation réservoir, ciblage forage)
```

### Ce que les agents couvrent

| Agent | Entrée | Sortie | Déblocage |
|-------|--------|--------|-----------|
| **`agent-gpu`** (Phase 1) | Fortran legacy | Fortran GPU + wrapper Python | Génération données IA, validation physique |
| **`agent-pipeline`** (Phase 2) | Fortran GPU | JAX différentiable | Entraînement surrogate, gradient-based inversion |

Les deux agents s'intègrent dans l'IDE (MCP) ou en CI/CD (CLI) — **l'ingénieur R&D garde la main**
sur le code généré via le mode Human-in-the-Loop avant compilation GPU.



## 1. 🏭 Le problème opérationnel

Les codes scientifiques HPC (sismique, météo, CFD) sont massivement écrits en Fortran des années 90 : **monolithiques, sans INTENT, avec COMMON blocks et état SAVE implicite.**

```fortran
! Exemple réel — seismic_CPML_2D (1 000 lignes, PROGRAM monolithique)
program seismic_CPML_2D_iso_second
  COMMON /grid/ dx, dy, NX, NY                    ! état global partagé
  double precision, save :: psi_dvx = 0.0          ! état caché entre appels
  ...
  do it = 1, NSTEP
    do j = 2, NY                                   ! kernel inline — non extractible
      do i = 2, NX
        sigma_xx(i,j) = sigma_xx(i,j) + ...
      end do
    end do
  end do
end program
```

**Le portage GPU manuel** d'un tel code prend **2–6 semaines** d'expertise HPC senior : extraction des kernels, annotation OpenACC, gestion des clauses `copyin`/`copy`, Cython wrapper, tests numériques.

**Cet agent automatise la transformation en une session (1/2 journée avec revue manuelle, 1 nuit en automatique)** :

| Étape | Entrée | Sortie | Gain |
|-------|--------|--------|------|
| **Phase 1** | Fortran CPU séquentiel | Fortran GPU (OpenACC) + wrapper Cython | ×10–100 sur GPU |
| **Phase 2** | Fortran GPU | JAX / XLA | Différentiable, fusionnable ML |

La démarche : partir du problème opérationnel concret (code sismique CPML),
en extraire des règles de transformation précises (INTENT, COMMON, SAVE, POINTER, types...),
puis généraliser à tout code Fortran HPC. Les [9 patterns documentés](#8--patterns-fortran--règles-de-transformation)
couvrent 95% des codes scientifiques rencontrés chez Total.

---

## 2. 🏗️ Architecture de la solution

```
📂 kernel.f90  (Fortran monolithique)
     │
     ▼ 🔍 parser          Loki AST — détecte INTENT, SAVE, COMMON, boucles, I/O
     │                    Zéro LLM — analyse déterministe
     │
     ▼ 🔧 extractor       LLM (1 appel) — extrait les boucles 2D en subroutines MODULE
     │                    Élimine COMMON blocks, expose SAVE comme INTENT(INOUT)
     │                    → module_kernels.f90  +  driver.f90
     │
     ▼ ✨ pure_elemental   Règles AST — annote PURE/ELEMENTAL (pas de LLM)
     │                    Valide : pas d'I/O, pas de SAVE, INTENT explicites
     │
     ▼ 🚀 openacc         LLM (1 appel driver) — !$acc parallel loop collapse(2)
     │                    !$acc data copyin/copy autour du time loop
     │
     ▼ 🐍 cython_wrapper  LLM (2 appels) — .pyx + kernel_c.h (iso_c_binding)
     │                    NumPy typed memoryviews, np.asfortranarray()
     │
     ▼ ✅ validation       gfortran × 2 flavors → nvfortran -acc (GPU)
     │                    Zéro LLM — compilation déterministe
     │
     📦 output/fortran_gpu/module_kernels_gpu.f90  +  output/cython/module.pyx
```

**Bilan LLM** : 4 appels maximum par pipeline (~2 min, ~$0.06 en tokens Mistral-Large-3).  
Loki fait le travail d'analyse AST de façon déterministe — le LLM n'intervient que là où
la compréhension sémantique est indispensable (extraction de kernels, génération d'interfaces).

---

## 3. ⚡ Démarrage rapide

### Pré-requis

- Python 3.12+, [uv](https://github.com/astral-sh/uv)
- `gfortran` (vérification syntaxe locale) : `brew install gcc`
- Endpoint Mistral OpenAI-compatible + API key (`.env`) — La Plateforme Mistral par défaut, ou vLLM/TGI auto-hébergé
- `nvfortran` (NVIDIA HPC SDK) ou nœud GPU souverain (Scaleway H100, GENCI, cluster on-prem) pour la compilation GPU

### Installation

#### Depuis PyPI (recommandé)

```bash
pip install fortranspire                # core + Loki (via loki-ifs) — analyze
pip install "fortranspire[gpu]"         # Phase 1 (LangChain + Cython)
pip install "fortranspire[mcp]"         # serveur MCP (HTTP/SSE)
pip install "fortranspire[all]"         # tout (Phase 1 + Phase 2 + MCP)
```

> **À propos de Loki** — ECMWF Loki n'est pas publié sur PyPI sous son nom
> d'origine (le nom `loki` y est pris par un package d'astronomie sans
> rapport). Nous maintenons une redistribution PyPI officielle sous le nom
> [`loki-ifs`](https://pypi.org/project/loki-ifs/), synchronisée avec
> [`ecmwf-ifs/loki@0.3.7`](https://github.com/maurinl26/loki-ifs).
> L'import Python reste `from loki import …` — `loki-ifs` est juste le
> nom de distribution PyPI. Source : https://github.com/maurinl26/loki-ifs.

#### Depuis les sources (pour développer)

```bash
git clone https://github.com/maurinl26/fortranspire
cd fortranspire
cp .env.example .env

# Choisir le profil d'installation selon l'usage :
uv sync                       # core + Loki (via loki-ifs) — analyze
uv sync --extra gpu           # Phase 1 : Fortran → GPU + Cython
uv sync --extra mcp           # serveur MCP (HTTP/SSE)
uv sync --extra jax           # Phase 2 : Fortran → JAX
uv sync --extra all           # tout (Phase 1 + Phase 2 + MCP)
```

### Connecter un endpoint Mistral (souverain)

L'agent appelle un endpoint Mistral OpenAI-compatible — **directement, sans passer par un hyperscaler**. Deux chemins recommandés selon le niveau de souveraineté visé.

#### A — La Plateforme Mistral (hébergement EU, opéré par Mistral AI)

```bash
# .env
MISTRAL_ENDPOINT="https://api.mistral.ai/v1"
MISTRAL_API_KEY="<clef_créée_sur_console.mistral.ai>"
MISTRAL_MODEL="mistral-large-latest"          # ou codestral-latest, mistral-nemo, ...

LLM_TEMPERATURE=0.0
LLM_TOP_P=0.9
LLM_NUM_PREDICT=2048
```

Clef à créer sur [console.mistral.ai](https://console.mistral.ai/) → *API Keys*. Facturation au token, infrastructure et données restent en Europe.

#### B — Auto-hébergement on-prem ou cloud privé (souveraineté complète)

Pour un cluster interne (Pangea, GENCI, datacenter privé), exposer le modèle via [vLLM](https://github.com/vllm-project/vllm), [TGI](https://github.com/huggingface/text-generation-inference) ou [Ollama](https://ollama.com/) — tous fournissent une API OpenAI-compatible.

```bash
# Exemple — vLLM avec Mistral-Large-Instruct-2407 (poids HuggingFace)
vllm serve mistralai/Mistral-Large-Instruct-2407 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4         # 4× A100 80GB
```

```bash
# .env côté agent
MISTRAL_ENDPOINT="http://<host>:8000/v1"
MISTRAL_API_KEY="dummy"            # vLLM ignore la clef sauf si --api-key est passé
MISTRAL_MODEL="mistralai/Mistral-Large-Instruct-2407"
```

#### Vérification

```bash
uv run python -c "from fortranspire.llm import get_llm; print(get_llm().invoke('ping').content)"
# → réponse du modèle, ou erreur d'auth / endpoint
```

> Tout endpoint exposant `POST {base}/chat/completions` au format OpenAI fonctionne. Seule la variable `MISTRAL_ENDPOINT` change selon le déploiement.

### Intégrations IDE (Claude Code & mistral-vibe)

`fortranspire` est utilisable depuis deux surfaces sans changement de code :

#### Claude Code (skill local, zéro plomberie MCP)

```bash
pip install fortranspire
mkdir -p .claude/skills
cp -r <fortranspire-repo>/skills/claude-code/fortranspire .claude/skills/
```

Recharge Claude Code → la slash-command `/fortranspire` apparaît et Claude
auto-invoque le skill sur tes fichiers `.f90`. Le skill couvre les 13 verbes
de la CLI et guide Claude vers `explain` (gratuit, sans LLM) **avant** tout
appel facturé à `gpu` ou `translate`. Détails dans
[docs/integrations/claude-code.md](docs/integrations/claude-code.md).

#### mistral-vibe (MCP HTTP/SSE)

```bash
pip install "fortranspire[mcp]"
fortranspire mcp --port 8000        # SSE sur http://<host>:8000/sse
```

Enregistre le serveur dans les paramètres MCP de mistral-vibe — neuf
outils sont exposés (`analyze_kernels`, `explain_port_cost`,
`build_call_graph`, `generate_docs`, `translate_kernel_gpu`,
`translate_kernel`, `profile_kernels`, `ask_agent`, `agent_status`).
Procédure complète dans
[docs/integrations/mistral-vibe.md](docs/integrations/mistral-vibe.md).

Même agent, deux LLM (Claude US / Mistral EU), trois postures de
souveraineté — détails dans la doc d'intégration.

### Usage CLI

```bash
# 🚀 Phase 1 — Fortran → Fortran GPU + Cython (recommandé)
uv run agent-gpu /path/to/kernel.f90

# 🔬 Phase 2 — Fortran → JAX (expérimental)
uv run agent-pipeline translate /path/to/kernel.f90

# 📊 Profil de performance
uv run agent-profile /path/to/kernel.f90
```

### Usage MCP (IDE — mode interactif)

```bash
docker compose up -d --build
```

```json
{
  "mcpServers": {
    "fortran-gpu-agent": {
      "transport": "sse",
      "url": "http://localhost:8000/sse"
    }
  }
}
```

Outils MCP : `translate_kernel_gpu`, `translate_kernel` (JAX), `ask_agent`, `profile_kernels`.

### Démarrer sur un repo Fortran (mode autonome)

Le serveur MCP peut traiter un repo Fortran de bout en bout sans intervention humaine — il extrait les kernels, annote PURE/ELEMENTAL, ajoute les directives OpenACC, génère le wrapper Cython (packaging `scikit-build`), et compile/valide tant qu'un environnement de build est disponible.

**Pré-requis d'environnement :**

| Ressource | Minimum (CPU) | Idéal (GPU) |
|-----------|---------------|-------------|
| Orchestration agent + Loki AST + LLM calls | ✅ requis | ✅ |
| `gfortran` — validation syntaxe + tests CPU | ✅ requis | ✅ |
| `nvfortran` (NVIDIA HPC SDK) — compilation `-acc -gpu=ccXX` | ❌ (skip GPU step) | ✅ requis |
| GPU device (T4, A100, …) — benchmark + nsys profile | ❌ | ✅ requis |

> En mode CPU seul, le pipeline va jusqu'à la génération du Fortran OpenACC + Cython et émet `output/compile_gpu.sh` — la compilation GPU est différée sur un nœud équipé (Scaleway H100, GENCI Jean-Zay, Pangea, cluster on-prem).

**Lancement autonome :**

```bash
export AGENT_INTERACTION_MODE=auto          # pas de pause Human-in-the-Loop
export AGENT_WORKSPACE=/path/to/fortran/repo
export AGENT_MAX_ITERATIONS=15

# Option A — un kernel à la fois (CLI)
uv run agent-gpu $AGENT_WORKSPACE/src/seismic_cpml_2d.f90

# Option B — sweep récursif sur tout le repo
find $AGENT_WORKSPACE -name "*.f90" -print0 | xargs -0 -n1 uv run agent-gpu

# Option C — MCP en daemon, l'IDE/CI pilote via translate_kernel_gpu
docker compose up -d --build
```

Les artefacts arrivent dans `output/` (voir ci-dessous). Le script
`output/compile_gpu.sh` est toujours généré — copie le contenu de
`output/` sur un nœud GPU (Scaleway H100, GENCI, Pangea, cluster on-prem)
et lance `bash compile_gpu.sh` pour compiler avec `nvfortran -acc`.

### Sorties

```
output/
├── fortran_gpu/
│   ├── kernel_pure.f90        ✨ PURE/ELEMENTAL annotated (hint sémantique)
│   ├── module_kernels_gpu.f90 🚀 OpenACC parallel loop (sans PURE)
│   ├── driver_gpu.f90         🔄 !$acc data region + time loop
│   ├── compile_gpu.sh         🛠️  Script de compilation GPU (détection arch auto)
│   └── validation.log         📋 Rapport gfortran + nvfortran
├── cython/
│   ├── module.pyx             🐍 Cython wrapper NumPy memoryviews
│   └── kernel_c.h             🔗 C header iso_c_binding
├── Makefile                   🔨 Build GPU
└── pyproject.toml             📦 Config nvfortran + Cython
```

---

## 4. ☁️ Déploiement (sovereign GPU)

`fortranspire` ne dépend d'aucun hyperscaler. La compilation GPU s'exécute
là où tu as du GPU disponible :

- **Local** — laptop ou workstation avec `nvfortran` (NVIDIA HPC SDK)
- **Cluster HPC** — GENCI Jean-Zay, Pangea (TotalEnergies), Météo-France
  Belenos, tout cluster Slurm avec NVIDIA HPC SDK chargé
- **Cloud souverain à la demande** — [Scaleway H100](https://www.scaleway.com/),
  OVHcloud GPU, Outscale (cloud souverain français certifié SecNumCloud)

Le pipeline génère toujours `output/compile_gpu.sh` qui détecte
l'architecture GPU et choisit les flags `nvfortran -acc -gpu=cc{XX}`
appropriés. Copie le dossier `output/` sur la cible et lance le script —
aucune infrastructure pré-provisionnée n'est requise.

Pour héberger l'agent en service (MCP HTTP/SSE accessible à toute une
équipe), le déploiement Docker + Apptainer est documenté dans
[docs/getting-started/installation.md](docs/getting-started/installation.md)
(Dockerfile, Dockerfile.hpc pour les sites HPC, apptainer.def, et
`docker compose up` pour démarrer le serveur MCP).

---

## 5. 🖥️ Intégration IDE (Human-in-the-Loop)

Le pipeline supporte un mode interactif où l'ingénieur peut réviser le code généré dans son IDE avant la suite du traitement.

```bash
export AGENT_INTERACTION_MODE=manual
uv run agent-gpu /path/to/kernel.f90
# → ouvre le fichier généré dans VSCode/PyCharm
# → appuyez ENTRÉE pour continuer après vos modifications
```

Ce mode est **seamless** pour une organisation comme TotalEnergies :
- Les ingénieurs HPC gardent la main sur les transformations critiques
- Les équipes DevOps utilisent le mode `auto` (CI/CD) pour les pipelines répétitifs
- Le serveur MCP permet d'invoquer le pipeline depuis n'importe quel IDE via l'assistant IA

Voir [TUTORIAL_IDE.md](TUTORIAL_IDE.md) pour le tutoriel complet.

---

## 6. 💰 FinOps — Estimation des coûts

| Composant | Coût unitaire | Par pipeline | Par mois (100 pipelines) |
|-----------|--------------|-------------|--------------------------|
| Mistral-Large (api.mistral.ai) | ~$3/1M tokens input | ~$0.06 (4 appels LLM) | ~$6 |
| CPU runner (orchestration) | ~$0.30/h | ~$0.03 (5 min) | ~$3 |
| GPU à la demande (H100 / A100 spot) | ~$0.80–1.50/h | ~$0.15 (10 min) | ~$15 |
| Stockage objet `output/` (S3/Scaleway) | ~$0.02/GB/mois | <$0.01 | <$1 |
| **Total PoC** | | **~$0.25** | **~$25/mois** |

*Comparaison : 1 semaine d'ingénieur HPC senior ≈ 5 000–8 000 €*  
*Retour sur investissement : > 99% de réduction du coût de portage GPU*

**Optimisation coûts** :
- Utiliser des VMs Spot (économie 60–80% vs On-Demand)
- Le mode `auto` (CI/CD) limite les runs aux commits validés — pas de runs manuels coûteux
- La réduction LLM (4 appels → 2 via délégation Loki) est en cours (voir roadmap)

---

## 7. 📊 Benchmark GPU — CPU vs GPU Speedup

Après génération du code GPU, compare les performances directement
sur ton nœud GPU :

```bash
# Compile et lance le binaire CPU de référence + le binaire GPU
cd output/
bash compile_gpu.sh            # détecte l'arch et compile avec nvfortran -acc
./kernel_cpu   < input.dat     # baseline gfortran
./kernel_gpu   < input.dat     # compilé avec -acc -gpu=cc{XX}

# Profiling NSight Systems (rapport détaillé des kernels GPU)
nsys profile --trace=cuda,nvtx ./kernel_gpu < input.dat
```

Le rapport `fortranspire bench output/` (voir [docs/getting-started/quickstart.md](docs/getting-started/quickstart.md))
agrège ces mesures sur plusieurs runs et détecte les régressions de
performances entre deux versions du pipeline.

**Résultats attendus — seismic_CPML_2D (NX=101, NY=641, NSTEP=2000)** :

| GPU | Speedup attendu | Temps CPU (ref) | Temps GPU |
|-----|----------------|-----------------|-----------|
| T4 (cc75) | 10–30× | ~8 s | ~0.3–0.8 s |
| A100 (cc80) | 50–100× | ~8 s | ~0.08–0.16 s |

**Ce que le script mesure** :
- Compilation CPU (nvfortran -O3, sans `-acc`) et GPU (nvfortran -acc -gpu=ccXX)
- Warmup GPU (1 run ignoré) puis benchmark chronométré
- Vérification numérique : comparaison des `velocnorm` CPU vs GPU (tolérance 1e-6)
- Rapport `benchmark.log` sur la VM distante

---

## 8. 🔬 Patterns Fortran & règles de transformation

Le LLM (Mistral-Large) et Loki (AST) analysent conjointement les patterns critiques dans le code source.

### 8.1 🔗 INTENT — la clé de tout le reste

INTENT définit le contrat de chaque argument. Sans INTENT explicite, ni OpenACC ni JAX ne peuvent fonctionner correctement.

```fortran
! ❌ Avant : INTENT implicite → ambiguïté totale
subroutine update_stress(vx, sigma_xx, NX)
  double precision vx(NX), sigma_xx(NX)   ! IN ou INOUT ?

! ✅ Après : INTENT explicite → contrat clair
subroutine update_stress(vx, sigma_xx, NX)
  integer,          intent(in)    :: NX
  double precision, intent(in)    :: vx(NX)
  double precision, intent(inout) :: sigma_xx(NX)
```

**Règles de transformation :**

| INTENT | OpenACC | JAX |
|--------|---------|-----|
| `IN` | `copyin(arr)` — copié une fois GPU avant la boucle | Argument immutable |
| `INOUT` | `copy(arr)` — synchro bidirectionnelle | Retourné par la fonction |
| `OUT` | `copyout(arr)` — rapatrié après calcul | Valeur de retour |
| Non déclaré | ⚠️ Loki infère depuis les lectures/écritures | ⚠️ Bloquant — doit être résolu |

---

### 8.2 🗃️ COMMON BLOCKS — état global à éliminer

COMMON est un bloc mémoire partagé entre toutes les routines — l'ennemi du GPU et de JAX.

```fortran
! ❌ Avant : COMMON block — état global implicite
COMMON /grid/ dx, dy, NX, NY
COMMON /fields/ vx(1000,1000), sigma_xx(1000,1000)

subroutine update_stress()
  ! vx et sigma_xx sont accessibles implicitement
  sigma_xx(i,j) = sigma_xx(i,j) + vx(i,j) * dx
end subroutine

! ✅ Après : arguments explicites dans un MODULE
MODULE seismic_kernels
contains
  subroutine update_stress(vx, sigma_xx, dx, NX, NY)
    integer,          intent(in)    :: NX, NY
    double precision, intent(in)    :: dx, vx(NX,NY)
    double precision, intent(inout) :: sigma_xx(NX,NY)
    sigma_xx(i,j) = sigma_xx(i,j) + vx(i,j) * dx
  end subroutine
END MODULE
```

**Règles de transformation :**
- **OpenACC** : impossible d'annoter `copyin`/`copy` sur des COMMON blocks → les extraire en arguments explicites
- **JAX** : pas de concept de variable globale mutable → tout doit être argument ou retour
- **Action** : l'agent `extractor` remplace chaque COMMON par des arguments `INTENT(IN/INOUT)` dans le MODULE généré

---

### 8.3 💾 SAVE — état persistant entre les appels

`SAVE` conserve la valeur d'une variable locale entre deux appels successifs à la même routine — un état caché.

```fortran
! ❌ Avant : variable SAVE = état caché entre appels
subroutine update_memory(dvx_dx)
  real, save :: psi_vx = 0.0    ! initialisée une fois, persiste
  psi_vx = b_x * psi_vx + a_x * dvx_dx
  dvx_dx = dvx_dx / K_x + psi_vx
end subroutine

! ✅ Après : état passé explicitement
subroutine update_memory(dvx_dx, psi_vx, b_x, a_x, K_x)
  real, intent(inout) :: psi_vx   ! état exposé, géré par l'appelant
  real, intent(inout) :: dvx_dx
  real, intent(in)    :: b_x, a_x, K_x
  psi_vx = b_x * psi_vx + a_x * dvx_dx
  dvx_dx = dvx_dx / K_x + psi_vx
end subroutine
```

**Règles de transformation :**
- **OpenACC** : une variable SAVE par thread GPU → race condition → doit devenir `INTENT(INOUT)` ou tableau indexé par thread
- **JAX** : `SAVE` brise la pureté fonctionnelle → `jax.lax.scan` gère l'état entre les itérations
- **Action** : l'agent `extractor` détecte les variables SAVE et les remonte comme arguments `INTENT(INOUT)`; les subroutines avec SAVE ne peuvent pas être annotées `PURE`

---

### 8.4 👉 POINTER — aliasing dangereux pour le GPU

Les pointeurs Fortran peuvent référencer des zones mémoire arbitraires — incompatible avec les clauses `!$acc data`.

```fortran
! ❌ Avant : pointeur avec aliasing potentiel
real, pointer :: field(:,:)
field => vx    ! ou sigma_xx selon le contexte

! ✅ Option A : remplacer par allocatable (si propriété claire)
real, allocatable :: field(:,:)
allocate(field(NX, NY))

! ✅ Option B : passer la cible directement comme argument INTENT(IN)
subroutine process(field, NX, NY)
  real, intent(inout) :: field(NX, NY)
```

**Règles de transformation :**
- **OpenACC** : les pointeurs fonctionnent si la cible est connue et unique, mais `!$acc data` exige un tableau concret de taille définie → préférer `allocatable`
- **JAX** : pas de pointeurs — remplacer par des slices d'arrays (`jnp.array[i:j]`)
- **Action** : Loki détecte les associations pointeur-cible; si la cible est statique, l'agent remplace par `allocatable` ou argument direct

---

### 8.5 🧱 Array of Structures → Structure of Arrays (AoS → SoA + collapse)

Les types dérivés Fortran créent des **Array of Structures** (AoS) : les champs d'un même élément sont contigus en mémoire. Sur GPU, tous les threads d'un warp accèdent au *même champ* sur des *éléments différents* — l'AoS force des accès non contigus (non-coalesced).

```fortran
! ❌ AoS — mauvais pour le GPU (accès non-coalesced)
type :: point_t
  real :: x, y, vx, vy
end type
type(point_t) :: particles(N)

do i = 1, N
  particles(i)%vx = particles(i)%vx + particles(i)%x * dt   ! thread i saute de 4 réels en 4 réels
end do

! ✅ SoA — optimal GPU (accès coalesced, column-major Fortran)
real :: x(N), y(N), vx(N), vy(N)

!$acc parallel loop
do i = 1, N
  vx(i) = vx(i) + x(i) * dt   ! threads contigus → un seul accès mémoire groupé
end do
```

**Pour les boucles 2D — `collapse(2)` :**

Sans `collapse`, seul le *j* extérieur est parallélisé (NY threads). Avec `collapse(2)`, les deux boucles fusionnent en NX×NY threads indépendants — utilisation GPU complète.

```fortran
! ❌ Sans collapse : NY threads seulement
!$acc parallel loop
do j = 2, NY
  do i = 2, NX                  ! boucle i reste séquentielle dans chaque thread
    sigma_xx(i,j) = sigma_xx(i,j) + ...
  end do
end do

! ✅ Avec collapse(2) : NX×NY threads — toutes les cellules en parallèle
!$acc parallel loop collapse(2) private(tmp_dx, tmp_dy)
do j = 2, NY
  do i = 2, NX
    sigma_xx(i,j) = sigma_xx(i,j) + ...
  end do
end do
```

**Règles de transformation :**

| Source | OpenACC | JAX |
|--------|---------|-----|
| `type(t) :: arr(N)` (AoS) | Séparer en tableaux scalaires SoA | `pytree` ou champs séparés `jnp.array` |
| Boucle 2D `do j; do i` indépendante | `!$acc parallel loop collapse(2)` | `jax.vmap` sur deux axes ou vectorisation implicite |
| Boucle 2D avec stencil `(i-1,j)` | `collapse(2)` OK si `i-1` vient d'un tableau déjà sur GPU | Même — JAX accède `a[i-1]` en slice |

> ⚠️ En Fortran column-major, la dimension **i** varie le plus vite en mémoire. Pour un accès coalesced GPU, la boucle intérieure doit itérer sur **i** (dimension 1) — c'est le cas dans les stencils FD classiques.

---

### 8.6 🔗 Dépendances imbriquées — boucles non-parallélisables

Une boucle est parallélisable seulement si chaque itération est **indépendante**. Les dépendances sur `i-1` dans la *même dimension* brisent ce principe.

```fortran
! ✅ Cas 1 — Stencil FD (dépendance sur i-1 d'un AUTRE tableau) → parallélisable
!$acc parallel loop collapse(2)
do j = 2, NY
  do i = 2, NX
    vx(i,j) = vx(i,j) + (sigma_xx(i,j) - sigma_xx(i-1,j)) / dx  ! sigma_xx est en lecture seule
  end do
end do

! ❌ Cas 2 — Dépendance récurrente sur le même tableau → NON parallélisable
do i = 2, N
  a(i) = coeff * a(i-1) + source(i)   ! a(i) dépend de a(i-1) calculé au tour précédent

! ❌ Cas 3 — Time loop (dépendance temporelle) → séquentiel sur host
do it = 1, NSTEP
  call update_stress(...)    ! état it+1 dépend de l'état it
  call update_velocity(...)
end do
```

**Stratégies de transformation :**

| Type de dépendance | OpenACC | JAX |
|-------------------|---------|-----|
| **Stencil FD** `a(i,j) ← b(i-1,j)` (tableaux différents) | `!$acc parallel loop collapse(2)` ✅ | `jax.vmap` ou vectorisation implicite ✅ |
| **Récurrence** `a(i) = f(a(i-1))` (même tableau) | ❌ Non parallélisable — laisser séquentiel ou reformuler | `jax.lax.scan` ✅ |
| **Time loop** `u(t+1) = f(u(t))` | `!$acc data` autour du loop (kernels GPU, time loop sur host) | `jax.lax.scan` avec carry ✅ |
| **Réduction** `sum += a(i)` | `!$acc loop reduction(+:sum)` ✅ | `jnp.sum(a)` ✅ |

**Exemple complet — time loop vers JAX :**

```fortran
! Fortran : time loop séquentiel sur host, kernels GPU en parallèle
!$acc data copyin(lambda,rho) copy(vx,vy,sigma_xx)
do it = 1, NSTEP                                ! séquentiel host — dépendance temporelle
  call update_stress(vx, sigma_xx, ...)         ! 🚀 GPU kernel (collapse 2D)
  call update_velocity(sigma_xx, vx, ...)       ! 🚀 GPU kernel (collapse 2D)
end do
!$acc end data
```

```python
# JAX : time loop → jax.lax.scan (jit-compilé, différentiable)
def time_step(carry, _):
    vx, vy, sigma_xx, sigma_yy, sigma_xy = carry
    sigma_xx, sigma_yy = update_stress(vx, vy, sigma_xx, sigma_yy, ...)
    vx, vy = update_velocity(sigma_xx, sigma_yy, sigma_xy, vx, vy, ...)
    return (vx, vy, sigma_xx, sigma_yy, sigma_xy), None

# Lance NSTEP itérations en un seul appel XLA compilé
(vx_f, vy_f, *_), _ = jax.lax.scan(time_step, init_carry, xs=None, length=NSTEP)
```

**Avantage JAX :** `jax.lax.scan` est différentiable — `jax.grad(loss)(params)` propage le gradient à travers toutes les NSTEP itérations. Utile pour l'inversion sismique (FWI) ou l'entraînement d'un surrogate.

---

### 8.7 ⚡ ELEMENTAL et OpenACC — le bon pattern

Une procédure `ELEMENTAL` **ne peut pas** contenir de directive compute OpenACC (`!$acc parallel`, `!$acc kernels`) — même contrainte que `PURE`. Mais elle est *parfaite* pour `!$acc routine seq` : elle s'exécute séquentiellement dans chaque thread GPU, appelée depuis le `!$acc parallel loop` de la routine parente.

```fortran
! ✅ Pattern correct : ELEMENTAL + !$acc routine seq
ELEMENTAL function pml_update(psi, field_deriv, b, a, K) result(corrected)
  !$acc routine seq           ← autorisé dans ELEMENTAL (pas un compute construct)
  real(dp), intent(in) :: psi, field_deriv, b, a, K
  real(dp) :: psi_new, corrected
  psi_new   = b * psi + a * field_deriv
  corrected = field_deriv / K + psi_new
end function

! La boucle parallèle est dans la ROUTINE PARENTE — pas dans l'ELEMENTAL
subroutine update_velocity_x(vx, sigma_xx, psi_dvx, b_x, a_x, K_x, ...)
  !$acc parallel loop collapse(2) private(dvx_dx)
  do j = 2, NY
    do i = 2, NX
      dvx_dx   = (sigma_xx(i,j) - sigma_xx(i-1,j)) / dx
      dvx_dx   = pml_update(psi_dvx(i,j), dvx_dx, b_x(i), a_x(i), K_x(i))  ← appel GPU
      vx(i,j)  = vx(i,j) + dvx_dx * dt / rho(i,j)
    end do
  end do
  !$acc end parallel
end subroutine
```

**Résumé des combinaisons autorisées :**

| Procédure | `!$acc parallel loop` intérieur | `!$acc routine seq` | Appelée depuis GPU |
|-----------|--------------------------------|---------------------|--------------------|
| `SUBROUTINE` standard | ✅ | ✅ | Avec `!$acc routine` |
| `PURE SUBROUTINE` | ❌ (standard) | ✅ | Avec `!$acc routine seq` |
| `ELEMENTAL FUNCTION` | ❌ | ✅ ← **usage correct** | ✅ depuis parallel loop |
| `ELEMENTAL SUBROUTINE` | ❌ | ✅ | ✅ depuis parallel loop |

> 💡 **Règle** : l'`ELEMENTAL` est le bon candidat pour les calculs *ponctuels* (un point du stencil, une correction PML, un terme source). Le `!$acc parallel loop collapse(2)` reste dans la routine parente qui itère sur tous les points.

---

### 8.8 🎯 Types explicites — précision mixte interdite au compilateur

Fortran autorise les déclarations implicites et les promotions silencieuses. Sur GPU, cette ambiguïté se paye en instructions fp32/fp64 mélangées et en conversions coûteuses entre registres.

**Règle absolue avant traduction : `IMPLICIT NONE` + types avec précision explicite.**

```fortran
! ❌ Avant : précision laissée au compilateur
REAL dx, dy                      ! 32 ou 64 bits selon -r8 / -fdefault-real-8 ?
DOUBLE PRECISION vx(NX, NY)      ! portable mais stylistiquement incohérent
REAL*8 sigma_xx(NX, NY)          ! extension non-standard (GCC/Intel only)
INTEGER NX                       ! OK — entiers 32 bits par défaut

! ✅ Après : précision déclarée explicitement via paramètre de KIND
integer, parameter :: dp = selected_real_kind(15, 307)   ! IEEE 754 double (64-bit)
integer, parameter :: sp = selected_real_kind(6,  37)    ! IEEE 754 single (32-bit)

real(dp) :: dx, dy               ! 64-bit partout — cohérent avec nvfortran -acc
real(dp) :: vx(NX, NY)
real(dp) :: sigma_xx(NX, NY)
integer  :: NX, NY               ! entier 32-bit — correct
```

**Précision mixte — quand c'est voulu :**

Sur A100, les opérations fp32 sont 2× plus rapides que fp64. Certains codes hybrides peuvent utiliser sp pour les arrays de travail et dp pour l'accumulation :

```fortran
! ✅ Précision mixte EXPLICITE — le compilateur n'infère rien
real(sp), intent(in)    :: source_term(NX, NY)  ! entrée basse précision (capteurs)
real(dp), intent(inout) :: accumulated(NX, NY)  ! accumulation haute précision

! Conversion explicite obligatoire (ne pas laisser le compilateur promouvoir silencieusement)
accumulated(i,j) = accumulated(i,j) + real(source_term(i,j), dp)
```

**Règles de transformation :**

| Pattern source | Transformation | Note |
|---------------|----------------|------|
| `REAL x` | `real(dp) :: x` | Supposer dp sauf indication contraire |
| `DOUBLE PRECISION x` | `real(dp) :: x` | Normaliser le style |
| `REAL*8 x` | `real(dp) :: x` | Extension non-standard → portable |
| `REAL*4 x` | `real(sp) :: x` | Explicite si voulu |
| `COMPLEX x` | `complex(dp) :: x` | Viscoélastique, acoustique complexe |
| Promotion implicite | `real(x, dp)` explicite | Jamais laisser `x + 1.0` si `x` est dp |
| Littéraux | `1.0_dp` au lieu de `1.0d0` | Cohérent avec KIND parameter |

**Pour JAX :** `jnp.float64` par défaut, forcer avec `jax.config.update("jax_enable_x64", True)`. Précision mixte possible avec `x.astype(jnp.float32)` explicite.

---

### 8.9 🔧 Flags logiques USE_xx → directives de compilation

Les codes scientifiques Fortran utilisent souvent des `LOGICAL PARAMETER` comme interrupteurs de fonctionnalités :

```fortran
LOGICAL, PARAMETER :: USE_PML        = .TRUE.
LOGICAL, PARAMETER :: USE_ATTENUATION = .FALSE.
LOGICAL, PARAMETER :: SAVE_SNAPSHOTS  = .TRUE.
```

**Problème GPU :** même si ces constantes sont évaluées à la compilation, les branches `if (USE_PML)` dans un `!$acc parallel loop` génèrent du code mort que *certains* compilateurs n'éliminent pas proprement → warp divergence potentielle.

**Transformation recommandée — CPP preprocessor :**

```fortran
! kernel.F90  (extension .F90 = preprocessing automatique avec nvfortran/gfortran)

#ifdef USE_PML
  ! Correction mémoire PML — compilé seulement si -DUSE_PML
  memory_dvx_dx(i,j) = b_x(i) * memory_dvx_dx(i,j) + a_x(i) * dvx_dx
  dvx_dx = dvx_dx / K_x(i) + memory_dvx_dx(i,j)
#endif
#ifdef USE_ATTENUATION
  sigma_xx(i,j) = sigma_xx(i,j) - tau_sigma * memory_sigma(i,j)
#endif
```

```bash
# Compilation avec les features activées
nvfortran -acc -gpu=cc80 -cpp \
  -DUSE_PML \
  -o seismic_gpu kernel.F90

# Version sans PML (benchmark comparatif)
nvfortran -acc -gpu=cc80 -cpp \
  -o seismic_gpu_nopml kernel.F90
```

**Équivalences multi-cibles :**

| Source Fortran | OpenACC / nvfortran | JAX |
|---------------|---------------------|-----|
| `LOGICAL, PARAMETER :: USE_PML = .TRUE.` | `#define USE_PML` → `-DUSE_PML` | `USE_PML = True` (constante Python) |
| `if (USE_PML) then ... end if` | `#ifdef USE_PML ... #endif` | `if USE_PML: ...` (évalué au jit-trace) |
| `if (USE_PML) then ... end if` dans parallel loop | `#ifdef` → **dead code éliminé** | `jax.lax.cond(USE_PML, f_pml, f_nopml, args)` si différentiable |
| Flag multi-valeur `INTEGER, PARAMETER :: SCHEME = 2` | `#if SCHEME == 2 ... #endif` | `if SCHEME == 2: ...` au trace time |

```python
# JAX — les flags Python sont évalués au moment du jit-trace, pas à l'exécution
USE_PML = True

@jax.jit
def update_velocity(vx, sigma_xx, psi_dvx, ...):
    dvx_dx = (sigma_xx[i,j] - sigma_xx[i-1,j]) / dx
    if USE_PML:               # ← évalué UNE FOIS au jit, pas à chaque itération
        psi_dvx = b_x * psi_dvx + a_x * dvx_dx
        dvx_dx  = dvx_dx / K_x + psi_dvx
    return vx + dvx_dx * dt / rho[i,j], psi_dvx

# Si USE_PML doit être différentiable → jax.lax.cond
dvx_dx, psi = jax.lax.cond(
    use_pml_flag,
    lambda args: pml_correction(*args),
    lambda args: (args[0], args[1]),
    (dvx_dx, psi_dvx),
)
```

> ⚠️ **Action agent** : Loki détecte les `LOGICAL PARAMETER` avec pattern `USE_*` ou `APPLY_*`. L'agent `extractor` les convertit en blocs `#ifdef` dans le fichier `.F90` généré et documente les flags actifs dans un header.

---

### 8.10 🌐 MPI Halo Exchange → GHEX (GPU-to-GPU)

> ⚠️ **Scope Phase 3 — non implémenté.** Ce pattern est documenté ici pour la planification.

Les codes MPI multi-domaines échangent des **halos** (bandes fantômes) entre processus à chaque pas de temps. Dans le schéma classique, ces échanges passent par la mémoire CPU — même si les arrays sont sur GPU :

```
GPU (proc 0)          CPU                GPU (proc 1)
   vx_local  ──acc update host──►  vx_host  ──MPI_Send──►  vx_host  ──acc update device──►  vx_local
   (device)          ↑                                                        ↓
                roundtrip CPU                                           roundtrip CPU
```

**Coût** : 2× PCIe transfers + latence MPI par pas de temps → annule une grande partie du gain GPU sur cluster multi-nœuds.

**Solution — GHEX (GridTools, ETH Zürich)** : échanges GPU-to-GPU directs via RDMA (NVLink ou InfiniBand + CUDA-aware MPI), sans roundtrip CPU.

```fortran
! ❌ Pattern actuel — halo exchange CPU (roundtrip coûteux)
!$acc update host(vx, vy)                              ! GPU → CPU
call MPI_Sendrecv(vx_send, ..., vx_recv, ..., MPI_COMM_WORLD, ...)
!$acc update device(vx, vy)                            ! CPU → GPU

! ✅ Pattern GHEX — halo exchange GPU-to-GPU (Phase 3)
! GHEX gère l'échange sur le device directement
call ghex_exchange(vx_field, vy_field, context)        ! RDMA GPU-to-GPU
! Pas de roundtrip CPU — les kernels suivants voient les halos à jour sur device
```

```python
# Côté Python/Cython — interface GHEX (Phase 3)
import ghex

ctx     = ghex.context(MPI.COMM_WORLD, thread_safe=False)
pattern = ghex.structured_pattern(ctx, domain, halo_width=1)

# Dans le time loop — échange GPU-to-GPU transparent
pattern.exchange(vx_field, vy_field).wait()
update_stress(vx, vy, sigma_xx, ...)
```

**Règles de transformation :**

| Pattern source | OpenACC + MPI (Phase 1) | OpenACC + GHEX (Phase 3) |
|---------------|------------------------|--------------------------|
| `MPI_Sendrecv` après `update_stress` | `!$acc update host` + MPI + `!$acc update device` | `ghex.exchange().wait()` |
| Arrays halo partagés | `INTENT(INOUT)` + sync CPU | `INTENT(INOUT)` + sync GPU |
| Overlap compute/comm | ❌ Séquentiel | ✅ `exchange()` asynchrone |

**Gain attendu** : réduction des communications de 3–10× sur cluster InfiniBand multi-GPU.

---

### 8.11 📡 I/O Fortran → xarray / zarr + DLPack

> ⚠️ **Scope Phase 4 — non implémenté.** Ce pattern est documenté ici pour la planification.

Les I/O Fortran classiques (`WRITE`, `OPEN`, PostScript) produisent des fichiers binaires propriétaires ou texte incompatibles avec l'écosystème data science moderne.

**Problème :** les codes seismiques écrivent des images `.pnm` et des sismogrammes `.dat` — illisibles directement par Pandas, xarray, ou les outils de visualisation cloud.

#### A — DLPack : zéro-copie entre Fortran GPU et Python

DLPack est un protocole de partage de tenseurs GPU entre frameworks (CUDA, JAX, PyTorch, CuPy) **sans copie mémoire**. Le wrapper Cython exposera les arrays GPU directement via DLPack :

```python
# Phase 1 actuelle — copie CPU nécessaire
vx_np = np.asfortranarray(vx)              # copie GPU → CPU → NumPy

# Phase 4 cible — zéro-copie via DLPack
from __dlpack__ import from_dlpack
import cupy as cp

vx_gpu = from_dlpack(seismic_module.vx_dlpack())   # vue DLPack directe sur la mémoire GPU
vx_jax = jax.dlpack.from_dlpack(vx_gpu)            # JAX array sans copie
vx_cp  = cp.from_dlpack(vx_gpu)                    # CuPy array sans copie
```

```fortran
! Côté Fortran — exposition du pointeur device via iso_c_binding (Phase 4)
function vx_device_ptr(vx) result(ptr) bind(C, name="vx_device_ptr")
  use iso_c_binding
  real(dp), device, intent(in) :: vx(:,:)    ! attribut device (nvfortran)
  type(c_ptr) :: ptr
  ptr = c_loc(vx)
end function
```

#### B — Sorties xarray / zarr (remplace WRITE / PostScript)

```python
# ❌ Actuel — fichiers binaires Fortran + PostScript
! WRITE(unit=27,...) image_data_2D    → fichiers .pnm
! WRITE(unit=11,...) sisvx(it, irec)  → fichiers .dat

# ✅ Phase 4 — sorties xarray/zarr cloud-native
import xarray as xr, zarr, numpy as np

# Construire un Dataset géophysique avec coordonnées
ds = xr.Dataset(
    {
        "vx":       (["x", "z", "time"], vx_history),      # champ de vitesse
        "sigma_xx": (["x", "z", "time"], stress_history),  # contrainte normale
        "seismo_x": (["receiver", "time"], sisvx),          # sismogrammes
    },
    coords={
        "x":    np.arange(NX) * DELTAX,
        "z":    np.arange(NY) * DELTAY,
        "time": np.arange(NSTEP) * DELTAT,
    },
    attrs={"source_x": ISOURCE * DELTAX, "source_z": JSOURCE * DELTAY},
)

# Écriture zarr — compatible stockage objet (S3, Scaleway, OVHcloud), Pangeo, Dask
ds.to_zarr("s3://seismic-results/run_001.zarr", mode="w")

# Lecture et visualisation directe sans conversion
import hvplot.xarray
ds["vx"].isel(time=100).hvplot(x="x", y="z", cmap="seismic")
```

**Règles de transformation :**

| Pattern source Fortran | Phase 1 (Cython) | Phase 4 (xarray/zarr) |
|------------------------|-----------------|----------------------|
| `WRITE(unit,...) field(NX,NY)` | NumPy array in-memory | `xr.DataArray` avec coords géo |
| `OPEN / WRITE / CLOSE` fichier `.dat` | Fichier texte Python | Zarr dataset sur stockage objet (S3 / Scaleway / OVHcloud) |
| Fichier image `.pnm` (PostScript) | Matplotlib imshow | hvPlot interactif / GeoViews |
| Sismogramme `.dat` par capteur | NumPy array | `xr.DataArray` indexé par receiver |
| Snapshot tous les N pas | Array 3D accumulé | Zarr avec append en streaming |

---

## 9. 📖 Philosophie de transformation

La chaîne de transformation suit une logique d'**activation progressive** : chaque étape rend la suivante possible. Ce n'est pas une séquence arbitraire.

```
Fortran (monolithique)
    │
    ▼  [extractor — LLM]
Fortran MODULAIRE (subroutines avec INTENT explicites)
    │  COMMON blocks → arguments MODULE
    │  SAVE → INTENT(INOUT) explicites
    │  INTENT implicite → INTENT(IN/OUT/INOUT)
    │
    ▼  [pure_elemental — règles AST, zéro LLM]
Fortran PUR (PURE/ELEMENTAL — fonctions sans effets de bord)
    │  hint sémantique : no I/O, no SAVE, INTENT stricts
    │
    ▼  [openacc — LLM driver + regex kernels]
Fortran GPU (OpenACC !$acc parallel loop — exécution sur A100)
    │  retire PURE, ajoute !$acc parallel loop collapse(2) + private()
    │  !$acc data copyin/copy autour du time loop dans le driver
    │
    ▼  [cython_wrapper — LLM]
Python/Cython (interface NumPy memoryviews — appelable depuis Python)
    │  iso_c_binding → cdef extern "kernel_c.h"
    │  cpdef + np.float64_t[:,::1] — zéro copie, column-major
    │
    ▼  [Phase 2, future]
JAX (jit/vmap — différentiable, fusionnable avec ML)
```

### Pourquoi cette séquence ?

**Étape 1 — Extraction : monolithique → modulaire**

Les codes comme `seismic_CPML_2D` sont des `PROGRAM` monolithiques — les boucles FD sont inline, sans subroutines ni INTENT.

- Sans INTENT explicites → impossible de déterminer `copyin` (lecture seule) vs `copy` (modifié in-place)
- Sans subroutines séparées → OpenACC ne peut pas cibler les bonnes boucles
- Sans MODULE → Cython ne peut pas générer de `cdef extern` propre

**Étape 2 — PURE/ELEMENTAL : effets de bord → fonctions pures**

| Propriété | Importance GPU | Importance JAX |
|-----------|---------------|----------------|
| Pas d'I/O | Les I/O ne s'exécutent pas sur device | idem |
| Pas de SAVE | Pas d'état caché → threads indépendants | Requis pour `jit` |
| INTENT explicite | Détermine `copyin` vs `copy` | Détermine les arguments JAX |
| Déterminisme | Résultat identique quel que soit l'ordre des threads | Requis pour `vmap` |

**Étape 3 — OpenACC : pattern complet pour stencil FD 2D**

```fortran
! Kernel — !$acc parallel loop collapse(2), PURE retiré
subroutine update_velocity_x(vx, sigma_xx, sigma_xy, rho, DELTAX, DELTAY, DELTAT, NX, NY)
  real(dp), intent(in)    :: sigma_xx(NX,NY), sigma_xy(NX,NY), rho(NX,NY)
  real(dp), intent(inout) :: vx(NX,NY)
  real(dp), intent(in)    :: DELTAX, DELTAY, DELTAT
  integer,  intent(in)    :: NX, NY
  real(dp) :: value_dsigma_xx_dx, value_dsigma_xy_dy   ! scalaires → private()

  !$acc parallel loop collapse(2) private(value_dsigma_xx_dx, value_dsigma_xy_dy)
  do j = 2, NY
    do i = 2, NX
      value_dsigma_xx_dx = (sigma_xx(i,j) - sigma_xx(i-1,j)) / DELTAX
      value_dsigma_xy_dy = (sigma_xy(i,j) - sigma_xy(i,j-1)) / DELTAY
      vx(i,j) = vx(i,j) + (value_dsigma_xx_dx + value_dsigma_xy_dy) * DELTAT / rho(i,j)
    enddo
  enddo
  !$acc end parallel
end subroutine

! Driver — !$acc data UNE SEULE FOIS avant les 2000 pas de temps
!$acc data copyin(lambda,mu,rho,b_x,a_x,K_x,...) &
!$acc      copy(vx,vy,sigma_xx,sigma_yy,sigma_xy,memory_dvx_dx,...)
do it = 1, NSTEP
  call update_stress_xx_yy(...)
  call update_velocity_x(...)
  if (mod(it, IT_DISPLAY) == 0) then
    !$acc update host(vx, vy)    ! rapatrier sur CPU pour affichage seulement
    print *, 'velocnorm =', maxval(sqrt(vx**2 + vy**2))
  endif
enddo
!$acc end data
```

Gain attendu : NX=101, NY=641, NSTEP=2000 → ~10s CPU → ~0.1s A100 (×100).

**Étape 4 — Cython → Python sans copie**

```python
import numpy as np
import seismic_cpml_2d_gpu as gpu_module

vx = np.asfortranarray(np.zeros((NX, NY)))   # layout column-major = Fortran
gpu_module.update_velocity_x(vx, sigma_xx, sigma_xy, rho, ...)
# Typed memoryviews = accès direct au buffer NumPy, zéro copie
```

**Phase 2 — JAX : les subroutines PURE deviennent des fonctions JAX directement**

| Fortran PURE | JAX équivalent |
|-------------|---------------|
| `PURE subroutine f(a, b, c_inout)` | `@jax.jit def f(a, b) -> c` |
| `INTENT(IN)` | Argument JAX (immutable) |
| `INTENT(INOUT)` | Valeur retournée |
| `do i,j` indépendants | `jax.vmap` ou vectorisation implicite |
| `do it` (time loop avec état) | `jax.lax.scan` avec carry |
| `ELEMENTAL function f(x)` | `jax.vmap(f, in_axes=0)` |

La traduction Fortran PURE → JAX est **mécanique** : les subroutines PURE sont des fonctions mathématiques pures — exactement ce que JAX compile en XLA.

### Déploiement GPU — workflow complet

```bash
# 1. Sur ton poste, génère le pipeline (CPU only)
fortranspire gpu /path/to/kernel.f90          # → output/

# 2. Copie output/ sur un nœud GPU (Scaleway H100, GENCI, Pangea, cluster on-prem)
rsync -a output/ user@<gpu-node>:~/seismic_gpu/

# 3. Sur le nœud GPU
ssh user@<gpu-node>
cd ~/seismic_gpu
bash compile_gpu.sh                    # détecte l'arch GPU + nvfortran -acc
nsys profile ./kernel_gpu < input.dat  # profiling NVIDIA NSight Systems

# Cluster HPC (GENCI Jean-Zay, Pangea, etc.) : charger nvhpc avant compile
# (sur Jean-Zay : module load nvhpc/24.1 && bash compile_gpu.sh)
```

---

## 10. 🗺️ Roadmap

La roadmap stratégique, les phases techniques et les surfaces de déploiement
sont maintenues dans [**ROADMAP.md**](ROADMAP.md). Vue rapide :

| Phase | Cible | Statut |
|-------|-------|--------|
| 1 — OpenACC + Cython | Fortran → GPU + wrapper Python | **Shipped (0.1.0)** |
| 1.5 — OpenMP target | Pragma multi-vendor (gfortran 13+, nvfortran, ifx) | Shipped (0.1.0) |
| 1.6 — GT4Py.next | DSL Python cartesian + unstructured (FVM / ICON) | Scoped — [#42](https://github.com/maurinl26/fortranspire/issues/42) |
| 2 — JAX | Kernels fonctionnels différentiables | Partial (0.2.x) |
| 3 — GHEX | Communications GPU-to-GPU MPI | Future |
| 4 — Modern I/O | xarray / zarr cloud-native | Future |
| 5 — Surrogates FNO | Operators neuronaux entraînés sur les sorties GPU | Future |

Travaux en cours suivis publiquement sur le [tracker](https://github.com/maurinl26/fortranspire/issues) — voir [`ROADMAP.md`](ROADMAP.md) pour le détail.

---

## 📦 Dépendances clés

| Paquet | Rôle |
|--------|------|
| `langgraph`, `langchain-openai` | Orchestration multi-agents |
| `loki-ifs>=0.3.7` | Parsing et transformation AST Fortran (redistribution PyPI d'ECMWF Loki) |
| `fastmcp` | Serveur MCP HTTP/SSE |
| `Cython`, `numpy` | Wrapper Python/Fortran |
| `gfortran` (brew install gcc) | Vérification syntaxe locale |
| `nvfortran` (NVIDIA HPC SDK) | Compilation Fortran GPU (-acc -gpu=cc80) |
| `jax[cpu]`, `flax`, `equinox` | Phase 2 — pipeline JAX (expérimental) |

---

## 🔎 Mode analyse seule (CI hook)

`agent-analyze` exécute uniquement l'étape Loki déterministe — **zéro
appel LLM, zéro fichier réécrit**. Pratique pour valider la conformité
GPU-ready d'un patch Fortran avant un appel LLM (coûteux) ou pour
verrouiller en CI :

```bash
# Texte humain (TTY coloré)
uv run agent-analyze src/

# SARIF — uploadable vers GitHub Code Scanning (annotations inline PR)
uv run agent-analyze --format sarif --output fortranspire.sarif src/

# Exit ≠ 0 si la sévérité atteint le seuil (par défaut: error)
uv run agent-analyze --fail-on warning src/
```

| Règle    | Sévérité  | Pattern détecté                                          |
| -------- | --------- | -------------------------------------------------------- |
| FORT001  | error     | I/O dans un kernel candidat (`PRINT`/`WRITE`/…)          |
| FORT002  | warning   | `SAVE` (état caché — à promouvoir en `INTENT(INOUT)`)    |
| FORT003  | warning   | `COMMON` block                                           |
| FORT004  | warning   | Dépendance loop-carried suspectée                        |
| FORT005  | warning   | `POINTER` (review pour data movement GPU)                |
| FORT006  | note      | `IMPLICIT NONE` manquant                                 |
| FORT007  | note      | `REAL`/`INTEGER` sans `KIND` explicite                   |
| FORT008  | note      | Derived `TYPE` (candidat SoA conversion)                 |
| FORT009  | error     | Échec de parsing Loki                                    |
| FORT010  | warning   | Aucun compilateur Fortran sur PATH (validation impossible) |
| FORT011  | warning   | Code `!$acc` détecté mais aucun compilateur OpenACC dispo  |

`agent-analyze` sonde aussi `PATH` à chaque exécution pour détecter
`gfortran` / `nvfortran` / `ifx` / `flang` / `lfortran`, leur version, et
leur capacité OpenACC. Le rapport final inclut le compilateur recommandé
pour le portage GPU :

```text
Toolchain:
  gfortran   13.2.0       family=gnu            openacc=experimental (-fopenacc)
  nvfortran  24.5         family=nvidia         openacc=native (-acc)
  → recommended for GPU port: nvfortran 24.5
```

**Intégration CI fournies** :

- **GitHub Actions** — [`.github/workflows/analyze.yml`](.github/workflows/analyze.yml)
  s'exécute sur chaque PR + push `main`, upload SARIF vers Code Scanning
  → annotations inline sur les PR (gratuit, marche sur fork).
- **Apptainer HPC** — [`Apptainer.analyze`](Apptainer.analyze) construit
  une image légère (pas de CUDA, pas de NVIDIA HPC SDK) pour exécuter
  l'analyzer dans un job Slurm / Jenkins on-prem (Pangea, GENCI, OVH) :
  `apptainer run fortranspire-analyze.sif --format sarif src/`.

---

## 📝 Documenter un code Fortran legacy (`agent-doc`)

Feature autonome — utilisable indépendamment du portage GPU :

```bash
# Docstrings Doxygen-style insérés directement dans le source
uv run agent-doc src/

# + site Sphinx auto-généré sous documentation/<projet>/
uv run agent-doc --sphinx src/

# Site uniquement (sources intacts)
uv run agent-doc --site-only src/

# Pas d'appel LLM — signatures seules, gratuit, idempotent
uv run agent-doc --no-llm src/
```

Deux niveaux de narration par routine : `@brief` (1 ligne, niveau
stakeholder) et `@details` (2-4 phrases, niveau développeur — INTENT,
invariants, pièges). Idempotent grâce au marker `@generated_by
fortranspire`. ~0,10 USD pour 50 routines via Codestral.

Voir [`docs/concepts/legacy-documentation.md`](docs/concepts/legacy-documentation.md).

---

## 🤝 Intégration Mistral

L'agent est **Mistral-first sans être Mistral-only** : pivot sur tout
endpoint OpenAI-compatible. Quatre chemins d'intégration sont documentés
dans [`docs/concepts/mistral-integration.md`](docs/concepts/mistral-integration.md) :

1. **LLM consumer** — La Plateforme Mistral comme moteur de génération
   (configuration `.env`, déjà en place).
2. **Modèle par étape** — `MISTRAL_MODEL_REASONING` (Mistral-Large pour
   extractor + openacc) et `MISTRAL_MODEL_CODE` (Codestral pour le
   wrapper Cython). Coût divisé par ~2, qualité préservée sur les étapes
   sémantiques.
3. **MCP provider** — le serveur `run-mcp` (FastMCP SSE) est consommable
   par Le Chat, l'API Mistral Agents, Cursor, Claude Desktop, …
4. **Connector directory Le Chat** — manifest prêt dans
   [`integration/le-chat-connector.json`](integration/le-chat-connector.json),
   soumission gated sur un endpoint MCP public stable (cf roadmap deploy).

> Aucune variable `AZURE_*` n'est lue par l'agent — la dépendance
> hyperscaler a été retirée au commit `ccfe221`.

Un smoke-test de la clef Mistral et du wiring Agents API est disponible :
[`examples/mistral_agents_api_smoke_test.py`](examples/mistral_agents_api_smoke_test.py).

---

## 📚 Documentation, citation, contribution

- **Documentation Sphinx** : <https://fortranspire.readthedocs.io>
- **Paper JOSS** : [`paper.md`](paper.md) (rendu en PDF par le workflow
  [`draft-paper.yml`](.github/workflows/draft-paper.yml) à chaque push)
- **Release Zenodo** : chaque tag Git tiré sur GitHub mint un DOI citable
  via l'intégration GitHub ↔ Zenodo (métadonnées dans [`.zenodo.json`](.zenodo.json))
- **Comment contribuer** : [`CONTRIBUTING.md`](CONTRIBUTING.md)
- **Sécurité / divulgation responsable** : [`SECURITY.md`](SECURITY.md)
- **Code of Conduct** : [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
  (Contributor Covenant 2.1)

### Citer ce projet

Si `fortranspire` vous a été utile dans un travail académique ou
industriel, merci de citer la release Zenodo (DOI auto-généré au prochain
tag) et, à publication, le paper JOSS — voir [`paper.md`](paper.md) pour
les métadonnées et [`paper.bib`](paper.bib) pour les références utilisées.

---

## 📜 Licence

Apache License 2.0 — voir [LICENSE](LICENSE).
