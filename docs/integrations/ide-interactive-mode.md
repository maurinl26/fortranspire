# 🖥️ Tutoriel : Mode Interactif (Human-in-the-Loop)

Ce tutoriel explique comment utiliser le mode interactif du pipeline **Fortran → GPU**
pour réviser le code généré directement dans votre IDE avant la suite du traitement.

## 1. Pourquoi le mode interactif ?

Le pipeline génère automatiquement :
- `output/fortran_gpu/module_kernels_gpu.f90` — kernels avec `!$acc parallel loop`
- `output/fortran_gpu/driver_gpu.f90` — driver avec `!$acc data` region
- `output/cython/module.pyx` — wrapper Python/Cython

Dans la plupart des cas, le code généré est correct. Mais sur les kernels complexes
(PML multi-couches, atténuation viscoélastique, grilles AMR), l'expertise humaine
permet d'affiner les clauses OpenACC avant de lancer la compilation GPU.

## 2. Activation du mode

```bash
export AGENT_INTERACTION_MODE=manual
```

Par défaut, le pipeline tourne en mode `auto` (CI/CD — aucune interaction).

## 3. Lancement du pipeline Phase 1

```bash
uv run agent-gpu /path/to/kernel.f90
```

Le pipeline s'arrête après chaque étape clé et ouvre le fichier généré dans votre éditeur
par défaut (VSCode, PyCharm, vim selon `$EDITOR`).

## 4. Déroulement de l'interaction

1. **Extraction** : l'agent extrait les kernels Fortran en subroutines MODULE.
   → ouvre `output/fortran_gpu/module_kernels.f90` dans l'IDE

2. **Révision** : vérifiez les INTENT, les bornes de boucles, les arguments manquants.
   Modifiez directement dans l'IDE si nécessaire.

3. **Validation** : revenez dans le terminal. L'agent affiche :
   ```
   👉 [ACTION REQUIRED] Review the generated Fortran module.
      Press ENTER when done (or type 'skip' to keep the current version).
   ```

4. **Continuité** : appuyez sur **ENTRÉE**. L'agent re-lit votre fichier modifié
   et l'utilise pour les étapes suivantes :
   - Annotation PURE/ELEMENTAL
   - Insertion `!$acc parallel loop`
   - Génération du wrapper Cython
   - Validation gfortran × 2 flavors

## 5. Intégration MCP (IDE natif)

Pour une intégration plus profonde, le serveur MCP expose les outils du pipeline
directement dans votre assistant IA (Claude, Copilot) :

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

Invocations disponibles depuis l'IDE :
- `translate_kernel_gpu` — Phase 1 : Fortran → GPU + Cython
- `translate_kernel` — Phase 2 : Fortran → JAX (expérimental)
- `profile_kernels` — analyse performance

## 6. Mode CI/CD (automatique)

Pour désactiver l'interaction et revenir au mode pipeline automatique :

```bash
export AGENT_INTERACTION_MODE=auto
uv run agent-gpu /path/to/kernel.f90
```

Ce mode est recommandé pour les pipelines répétitifs sur un lot de fichiers.

> [!TIP]
> Le mode interactif est idéal pour les kernels complexes (PML, atténuation, multi-physique)
> où l'expertise HPC est nécessaire pour valider les clauses OpenACC avant la compilation GPU.
> Pour les codes simples (boucles FD standard), le mode `auto` produit un code correct directement.
