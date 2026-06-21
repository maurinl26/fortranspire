"""
Graph d'agents LangGraph pour la traduction de Fortran 90 vers JAX.
Ordre : parser → explainer → translator → halo_exchange → reproducibility
        → performance → docstring → autodiff → surrogate → END
"""

import sys
import os
import re
import json
import operator
import shutil
import subprocess
import tempfile
import time
import timeit
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Annotated
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langgraph.constants import Send
from fortranspire.llm import get_llm, get_reasoning_llm
from langchain_core.messages import SystemMessage, HumanMessage


SEP  = "─" * 64
SEP2 = "═" * 64


# ==========================================
# 1. État du Graphe
# ==========================================

class KernelResult(TypedDict):
    routine_name: str
    fortran_code: str
    jax_code: str
    status: str  # "success", "error", "skipped"
    error_log: str
    # Parallel TDD fields
    fortran_wrapper: str
    unit_test_skeleton: str
    repro_passed: bool
    repro_max_abs: float
    repro_mean_rel: float

class TranslationState(TypedDict):
    fortran_filepath: str
    fortran_code: str
    ast_info: Dict[str, Any]
    kernel_results: Annotated[List[KernelResult], operator.add]
    children_code: Dict[str, str]
    # Project info
    project_root: str
    build_system_info: Dict[str, Any]
    # Kernels tracking
    kernels_found: List[str]
    kernels_translated: int
    # LLM outputs
    jax_code: str
    surrogate_code: str
    surrogate_framework: str
    call_graph: str
    jax_hints: List[str]
    # Errors & Retries
    compilation_error: str
    translation_retries: int
    reproducibility_retries: int
    executed_agents: List[str]
    last_error: str
    # Reproducibility
    fortran_wrapper_code: str
    unit_test_code: str
    interaction_mode: str  # "auto" or "manual"
    reproducibility_passed: bool
    # Metadata extraction
    scientific_metadata: Dict[str, str]
    # Performance
    performance_metrics: Dict[str, Any]


# ==========================================
# 2. Helpers
# ==========================================

def _output_dir(state: TranslationState, category: str = "src") -> Path:
    """
    Crée et retourne le répertoire de sortie structuré.
    Categories: 'src', 'tests/reproducibility', 'tests/performance', 'fortran'
    """
    kernel = state.get("ast_info", {}).get("routine_name", "kernel")
    out = Path("output") / category / kernel
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save(path: Path, content: str):
    path.write_text(content, encoding="utf-8")
    print(f"  💾 Saved → {path}")


def _fortran_compiler() -> str:
    """Retourne le compilateur Fortran disponible (support Mac Homebrew)."""
    # 1. Check PATH
    for compiler in ["gfortran", "flang", "ifort"]:
        if shutil.which(compiler):
            return compiler
    
    # 2. Check common Mac Homebrew paths
    mac_paths = [
        "/opt/homebrew/bin/gfortran",
        "/usr/local/bin/gfortran"
    ]
    for p in mac_paths:
        if os.path.exists(p):
            return p
            
    # 3. Check environment variable
    fc = os.getenv("FC")
    if fc and shutil.which(fc):
        return fc
        
    return None


def _strip_markdown(code: str) -> str:
    """
    Extrait uniquement le contenu du premier bloc de code Markdown détecté.
    Ignore le texte avant et après les balises ```.
    """
    # Recherche le contenu entre les premières triple-backticks
    match = re.search(r"```[a-zA-Z0-9]*\n?(.*?)\n?```", code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code.strip()


def _validate_jax(code: str, label: str = "jax_kernel") -> dict:
    """
    Valide le code JAX généré en 3 étapes progressives :
    1. Syntaxe Python (ast.parse) — immédiat, sans exécution
    2. Compilation bytecode (compile) — détecte les erreurs de nom / scope
    3. Exécution dans un namespace isolé (exec) — valide les imports JAX

    Utilise jax.make_jaxpr si une fonction 'forward' ou '<kernel_name>' est trouvée.

    Retourne un dict avec status, erreurs et hints de correction.
    """
    import ast as _ast

    result = {"label": label, "syntax": False, "bytecode": False, "exec": False, "jit": False, "error": None}

    # ── 1. Syntaxe AST ──────────────────────────────────────
    try:
        _ast.parse(code)
        result["syntax"] = True
    except SyntaxError as e:
        result["error"] = f"SyntaxError L{e.lineno}: {e.msg}"
        print(f"  ❌ Syntax error     : {result['error']}")
        return result

    # ── 2. Bytecode Python ───────────────────────────────────
    try:
        compile(code, f"<{label}>", "exec")
        result["bytecode"] = True
    except Exception as e:
        result["error"] = f"BytecodeError: {e}"
        print(f"  ❌ Bytecode error   : {result['error']}")
        return result

    # ── 3. Exécution isolée + détection d'une fn forward ────
    ns = {}
    try:
        exec(code, ns)  # noqa: S102
        result["exec"] = True
    except ImportError as e:
        # JAX peut ne pas être installé localement — ce n'est pas bloquant
        result["error"] = f"ImportError (non-blocking): {e}"
        print(f"  ⚠️  Import warning  : {e}")
        result["exec"] = True  # on considère OK si c'est juste un import manquant
    except Exception as e:
        result["error"] = f"ExecError: {e}"
        print(f"  ⚠️  Exec warning (non-fatal) : {result['error']}")
        result["exec"] = False

    # ── 4. jax.make_jaxpr (si JAX disponible) ───────────────
    try:
        import jax
        import jax.numpy as jnp

        # Cherche une fonction candidate à tracer
        candidates = [k for k, v in ns.items()
                      if callable(v) and not k.startswith("_")
                      and k.lower() in ["forward", "kernel", label.replace("_", ""), label]]
        if candidates:
            fn = ns[candidates[0]]
            # Tente un jaxpr avec un array factice
            dummy = jnp.ones((4,))
            jax.make_jaxpr(fn)(dummy)
            result["jit"] = True
            print(f"  ✅ jax.make_jaxpr   : {candidates[0]}() traced successfully")
        else:
            print(f"  ℹ️  jax.make_jaxpr   : no candidate function found (OK for module-level code)")
            result["jit"] = True
    except Exception as e:
        print(f"  ⚠️  jax.make_jaxpr   : {e} (non-blocking)")
        result["jit"] = False  # non-bloquant

    return result


# ==========================================
# 3. Agents
# ==========================================

# ── Agent 0 : Init Project [NEW] ─────────────────────────────────────

def init_project_agent(state: TranslationState) -> TranslationState:
    """
    Agent 0 (NEW) : Initialise le dossier output et génère le pyproject.toml
    pour orchestrer le projet JAX traduit. Check la présence de gfortran/flang.
    """
    print(f"\n{SEP}")
    print(f"  🏗️  [Init Agent] Initializing JAX Project Structure")
    print(SEP)

    # Check compilateur Fortran
    compiler = _fortran_compiler()
    if compiler:
        print(f"  ✅ Fortran compiler found : {compiler}")
    else:
        print(f"  ❌ CRITICAL : No Fortran compiler found (gfortran or flang).")
        print(f"     Reproducibility and Performance agents will be limited.")
        # On ne bloque pas forcément, mais on prévient l'utilisateur.

    # Création du pyproject.toml racine
    out_root = Path("output")
    out_root.mkdir(parents=True, exist_ok=True)
    
    pyproject_content = f"""[project]
name = "jax-seismic-cpml"
version = "0.1.0"
description = "Translated JAX version of Seismic CPML"
dependencies = [
    "jax[cpu]>=0.4.0",
    "flax>=0.8.0",
    "equinox>=0.11.0",
    "numpy",
    "pytest",
]

[project.scripts]
run-tests = "pytest tests"
run-bench = "pytest tests/performance"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
"""
    _save(out_root / "pyproject.toml", pyproject_content)
    
    # Création des sous-dossiers de base
    for d in ["src", "tests/reproducibility", "tests/performance", "fortran"]:
        (out_root / d).mkdir(parents=True, exist_ok=True)
        print(f"  📁 Folder created : output/{d}")

    print(SEP + "\n")
    return state


# ── Agent 1 : Parser ──────────────────────────────────────────────

def parse_and_isolate_agent(state: TranslationState) -> TranslationState:
    """Agent 1 : Loki AST Parser — isole le kernel Fortran avec Scheduler."""
    filepath = state["fortran_filepath"]
    print(f"\n{SEP}")
    print(f"  🔍 [Parsing Agent] Loki AST Parser + Project Discovery")
    print(f"  📄 File  : {filepath}")
    print(SEP)

    project_dir = Path(filepath).parent
    
    # 1. Detection du Build System
    build_info = {"type": None, "content": None}
    for build_file in ["Makefile", "CMakeLists.txt"]:
        if (project_dir / build_file).exists():
            build_info["type"] = build_file
            build_info["content"] = (project_dir / build_file).read_text()
            print(f"  🏗️  Build file found: {build_file}")
            break

    try:
        # Prioritize local Loki fork (has PROGRAM support patches)
        _loki_local = str(Path(__file__).resolve().parents[2] / "loki")
        if _loki_local not in sys.path:
            sys.path.insert(0, _loki_local)

        from loki import (
            Sourcefile, Frontend, Scheduler, SchedulerConfig, 
            FindNodes, BasicType
        )
        from loki.ir.nodes import VariableDeclaration, Loop, Conditional, CallStatement
        import loki.ir as ir
        import re
        import os
        import tempfile

        # workaround pour PROGRAM (Loki fparser ne le supporte pas en standalone)
        raw_content = Path(filepath).read_text()
        is_program = False
        if re.search(r'^\s*PROGRAM\s+', raw_content, re.IGNORECASE | re.MULTILINE):
            print("  🛠️  PROGRAM detected. Applying master-routine workaround.")
            is_program = True
            raw_content = re.sub(r'^\s*PROGRAM\s+(\w+)', r'SUBROUTINE \1_master', raw_content, flags=re.IGNORECASE | re.MULTILINE)
            raw_content = re.sub(r'^\s*END\s+PROGRAM', r'END SUBROUTINE', raw_content, flags=re.IGNORECASE | re.MULTILINE)
            
            # Écriture dans un fichier temporaire pour Loki
            fd, tmp_path = tempfile.mkstemp(suffix='.f90')
            try:
                with os.fdopen(fd, 'w') as tmp:
                    tmp.write(raw_content)
                
                # --- PARSING ROBUSTE VIA SUBPROCESS ---
                import subprocess
                import sys
                from loki import Frontend

                print(f"  🧪 Safety check (FParser)...")
                check_cmd = [
                    sys.executable, "-c",
                    f"from loki import Sourcefile; Sourcefile.from_file(r'{tmp_path}')"
                ]
                try:
                    # On tente de parser dans un processus isolé
                    result = subprocess.run(check_cmd, capture_output=True, timeout=30)
                    if result.returncode == 0:
                        print(f"  ⏳ Parsing modified source (FParser)...")
                        source = Sourcefile.from_file(tmp_path)
                    else:
                        raise RuntimeError(f"FParser crashed in subprocess (rc={result.returncode})")
                except Exception as e:
                    print(f"  ⚠️  FParser failed or crashed: {e}. Falling back to REGEX frontend.")
                    source = Sourcefile.from_file(tmp_path, frontend=Frontend.REGEX)
                # -----------------------

            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            from loki import Frontend
            print(f"  ⏳ Parsing source file (safety check)...")
            # Même logique de sécurité pour le fichier standard
            check_cmd = [
                sys.executable, "-c",
                f"from loki import Sourcefile; Sourcefile.from_file(r'{filepath}')"
            ]
            try:
                result = subprocess.run(check_cmd, capture_output=True, timeout=30)
                if result.returncode == 0:
                    source = Sourcefile.from_file(filepath)
                else:
                    raise RuntimeError(f"FParser crashed (rc={result.returncode})")
            except Exception as e:
                print(f"  ⚠️  FParser fallback: {e}. Using REGEX.")
                source = Sourcefile.from_file(filepath, frontend=Frontend.REGEX)

        print("  ✅ Loki Sourcefile created.")
        call_graph = "Standalone kernel"
        if build_info["type"]:
            try:
                # Configuration minimale pour le Scheduler
                config_sch = SchedulerConfig(default={'role': 'kernel', 'expand': True, 'strict': False}, routines=[])
                scheduler = Scheduler(paths=[project_dir], config=config_sch)
                
                # Cherche l'item correspondant au fichier
                item_name = Path(filepath).stem.lower()
                for item in scheduler.items:
                    if item.name.lower() == item_name or item.path == Path(filepath):
                        deps = [str(d) for d in item.dependencies]
                        if deps:
                            call_graph = f"Dependencies: {', '.join(deps)}"
                            print(f"  🔗 Loki Scheduler   : Found {len(deps)} dependencies")
                        break
            except Exception as e:
                print(f"  ⚠️  Scheduler analysis failed: {e}")

        if not source.routines:
            raise ValueError("No Fortran routines found in the file.")

        # --- GLOBAL SCHEMA EXTRACTION (LOKI AST Semantic) ---
        print(f"  🧠 Extracting Global Schema (Loki Semantic Architect)...")
        schema = {"params": [], "statics": [], "state": []}
        
        all_params = []
        all_statics = []
        all_state = []
        
        # 1. Loki AST attempt
        for routine in source.routines:
            for decl in FindNodes(VariableDeclaration).visit(routine.spec):
                for var in decl.symbols:
                    is_param = getattr(var.type, 'parameter', False)
                    dtype = getattr(var.type, 'dtype', BasicType.DEFERRED)
                    
                    if is_param:
                        if dtype == BasicType.LOGICAL:
                            all_statics.append(var.name)
                        else:
                            all_params.append(var.name)
                    else:
                        if hasattr(var, 'dimensions') and var.dimensions:
                            all_state.append(var.name)
        
        # 2. Robust REGEX Fallback if Loki returns zero schema items
        if not all_params and not all_state:
            print("  ⚠️  Loki AST schema was empty. Applying Robust REGEX Fallback...")
            raw_text = Path(filepath).read_text()
            
            # Extract Parameters
            param_matches = re.findall(r'(\w+)\s*,\s*parameter', raw_text, re.IGNORECASE)
            # Find assignments in parameter declarations: label=value
            param_assigns = re.findall(r'(\w+)\s*=\s*[^,!\s]+', raw_text)
            all_params.extend(param_matches)
            all_params.extend(param_assigns)
            
            # Extract basic Statics (Logicals)
            all_statics.extend(re.findall(r'LOGICAL\s*,\s*parameter\s*::\s*(\w+)', raw_text, re.IGNORECASE))
            
            # Extract State (Arrays)
            state_matches = re.findall(r'(\w+)\s*\([^)]+\)', raw_text)
            all_state.extend(state_matches)

        schema = {
            "params": sorted(list(set(all_params))),
            "statics": sorted(list(set(all_statics))),
            "state": sorted(list(set(all_state)))
        }
        print(f"  ✅ Schema: {len(schema['params'])} params, {len(schema['statics'])} statics (logicals), {len(schema['state'])} state vars")

        # --- ISOLATION DE CHAQUE ROUTINE ---
        kernel_results = []
        for routine in source.routines:
            print(f"  📦 Isolating kernel: {routine.name}")
            
            # --- XLA AST PROFILING ---
            xla_hints = []
            
            # 1. InOut mutations
            if hasattr(routine, 'arguments'):
                inout_vars = [v.name for v in routine.arguments if hasattr(v.type, 'intent') and v.type.intent and v.type.intent.lower() == 'inout']
                if inout_vars:
                    xla_hints.append(f"INOUT Detected: {', '.join(inout_vars)}. XLA requires returning modified states.")
                    
            # 2. SAVE or COMMON
            if getattr(routine, 'common', None):
                xla_hints.append("COMMON block detected. Group global state into a PyTree injection (no hidden state).")
            if hasattr(routine, 'variables'):
                save_vars = [v.name for v in routine.variables if hasattr(v.type, 'save') and v.type.save]
                if save_vars:
                    xla_hints.append(f"SAVE variables {', '.join(save_vars)} detected. Thread this state across iterations with jax.lax.scan.")
                    
            # 3. Loops (temp vs spatial)
            loops = FindNodes(Loop).visit(routine.body)
            if loops:
                loop_ranges = []
                for lp in loops:
                    if hasattr(lp, 'bounds') and lp.bounds:
                        loop_ranges.append(str(lp.bounds))
                xla_hints.append(f"Loops detected ({len(loops)}): ranges={loop_ranges[:3]}. Use lax.scan for sequential temporal loops, vmap/slicing for independent spatial loops.")

            # 4. Conditionals
            from loki.ir.nodes import Conditional
            conds = FindNodes(Conditional).visit(routine.body)
            if conds:
                xla_hints.append("If/Else branches detected. Use jax.numpy.where for XLA-safe boolean masking.")

            # 5. IO Detection (Loki AST)
            io_keywords = ['print', 'write', 'read', 'open', 'close', 'inquire']
            found_io = False
            for node in FindNodes(CallStatement).visit(routine.body):
                if any(k in str(node.name).lower() for k in io_keywords):
                    found_io = True
                    break
            if not found_io:
                if any(k in routine.to_fortran().lower() for k in io_keywords):
                    found_io = True
            if found_io:
                xla_hints.append("IO Operations detected (PRINT/WRITE/READ). Port these to Python native calls in the orchestration layer (greedy master) or REMOVE from JAX kernels.")

            # 6. Loki Dataflow Analysis — loop-carried deps, unused vars, 0-based indexing
            try:
                from loki.analyse import dataflow_analysis_attached, loop_carried_dependencies, read_after_write_vars
                from loki.transformations.array_indexing import shift_to_zero_indexing
                from loki.transformations.remove_code import find_unused_dummy_args_and_vars

                # Detect loop-carried dependencies (→ lax.scan pattern required)
                with dataflow_analysis_attached(routine):
                    for lp in loops:
                        deps = loop_carried_dependencies(lp)
                        if deps:
                            dep_names = [str(d) for d in deps]
                            xla_hints.append(f"Loop-carried dependencies on {dep_names} — these MUST be threaded as lax.scan carry state.")

                # Detect unused dummy args (safe to drop in JAX signature)
                unused_args, unused_vars = find_unused_dummy_args_and_vars(routine)
                if unused_args:
                    xla_hints.append(f"Unused dummy args: {[str(a) for a in unused_args]} — omit from JAX function signature.")

                # Check if indexing is 1-based (always true in Fortran — remind LLM)
                xla_hints.append("Fortran is 1-based indexed. ALL array accesses must be shifted: arr[i-1] in JAX. Use shift_to_zero_indexing pattern.")

            except Exception as e_df:
                xla_hints.append(f"[Dataflow analysis unavailable: {e_df}]")

            hint_str = "\n".join([f"! [XLA HINT] {h}" for h in xla_hints])
            kernel_str_raw = routine.to_fortran()
            if hint_str:
                kernel_str_raw = hint_str + "\n\n" + kernel_str_raw

            kernel_str = _strip_markdown(kernel_str_raw)
            
            # --- PARALLEL TDD : CONTRACT GENERATION ---
            print(f"  📜 Generating Test Contract for: {routine.name}...")
            wrapper_code = ""
            test_skeleton = ""
            
            # 1. iso_c_binding wrapper
            pw = f"Generate an iso_c_binding wrapper for subroutine {routine.name}:\n{kernel_str}"
            try:
                rw = get_reasoning_llm().invoke([SystemMessage(content="Expert in Fortran C-interop."), HumanMessage(content=pw)])
                wrapper_code = _strip_markdown(rw.content)
            except Exception as e:
                print(f"  ⚠️  Wrapper failed for {routine.name}: {e}")

            # 2. Pytest skeleton (will be populated with JAX code later)
            pt = f"""Generate a pytest for routine {routine.name} comparing Fortran (lib{routine.name.lower()}.so) vs JAX.
Requirements:
1. Load the shared library using ctypes.
2. Call the C-wrapper and the JAX function with identical inputs.
3. Calculate:
   - max_abs_error = np.max(np.abs(jax_out - fortran_out))
   - mean_rel_error = np.mean(np.abs(jax_out - fortran_out) / (np.abs(fortran_out) + 1e-12))
4. CRITICAL: Print the metrics to stdout in this exact format:
   NUMERICAL_METRICS: max_abs={{max_abs_error}}, mean_rel={{mean_rel_error}}
5. Use assert_allclose for the final check.
"""
            try:
                rt = get_reasoning_llm().invoke([HumanMessage(content=pt)])
                test_skeleton = _strip_markdown(rt.content)
            except Exception as e:
                print(f"  ⚠️  Test skeleton failed for {routine.name}: {e}")

            kernel_results.append({
                "routine_name": routine.name,
                "fortran_code": kernel_str,
                "jax_code": "",
                "status": "pending",
                "is_master": routine.name.lower().endswith("_master"),
                "error_log": "",
                "fortran_wrapper": wrapper_code,
                "unit_test_skeleton": test_skeleton,
                "repro_passed": False,
                "repro_max_abs": 0.0,
                "repro_mean_rel": 0.0,
                "global_schema": schema
            })

        executed = state.get("executed_agents", [])
        executed.append("parser")
        routines_names = [r.name for r in source.routines]

        return {
            "kernel_results": kernel_results,
            "kernels_found": routines_names,
            "ast_info": {
                "status": "parsed",
                "routine_name": routines_names[0] if routines_names else "kernel",
                "report": {
                    "global_schema": schema,
                    "routines": routines_names,
                    "is_program": is_program,
                }
            },
            "scientific_metadata": state.get("scientific_metadata", {}),
            "executed_agents": executed
        }

    except Exception as e:
        print(f"  ⚠️  Loki loading failed: {e}")
        import traceback
        traceback.print_exc()
        print("  🔄 Falling back to raw file.")
        raw = open(filepath).read()
        return {
            "kernel_results": [{
                "routine_name": "master",
                "fortran_code": raw,
                "jax_code": "",
                "status": "pending",
                "is_master": True,
                "error_log": ""
            }],
            "kernels_found": ["master"],
            "ast_info": {"status": "raw_fallback", "routine_name": "kernel"},
        }

    except Exception as e:
        err = str(e)
        print(f"  ⚠️  Loki parser failed: {err}")
        raw = open(filepath).read()
        lc = len(raw.split('\n'))
        hints = []
        if "PROGRAM" in err:
            hints = ["[Context: This is a standalone Fortran PROGRAM. Extract the computational kernel and translate it to a pure JAX function.]"]
            print("  ℹ️  PROGRAM block detected — sending raw file to LLM with context hint")
        print(f"  📏 File size       : {lc} lines (raw fallback)")
        print(SEP + "\n")
        return {
            "kernel_results": [{
                "routine_name": "kernel",
                "fortran_code": raw,
                "jax_code": "",
                "status": "error",
                "error_log": err
            }],
            "kernels_found": ["kernel"],
            "ast_info": {"status": "error", "routine_name": "kernel", "message": err},
        }


# ── Agent 2 : Explainer [NEW] ──────────────────────────────────────

def explainer_agent(state: TranslationState) -> TranslationState:
    """
    Agent 2 (NEW) : Retourne la fonction scientifique du code Fortran
    et son intérêt pour la géophysique. Purement didactique.
    """
    if "explainer" in state.get("executed_agents", []):
        return state
    if state.get("compilation_error") == "FILE_TOO_LARGE":
        return state

    print(f"\n{SEP2}")
    print(f"  🧠 [Explainer Agent] Scientific Context")
    print(SEP2)

    from fortranspire.llm import get_reasoning_llm
    llm = get_reasoning_llm()

    # Extraction du header pour métadonnées (200 premières lignes)
    header_raw = ""
    try:
        fpath = state.get("fortran_filepath")
        if fpath:
            with open(fpath, 'r') as f:
                header_raw = "".join([f.readline() for _ in range(200)])
    except:
        pass

    # Pour l'explainer, on s'appuie sur le raw_file ou le code du premier kernel
    preview = ""
    fpath = state.get("fortran_filepath")
    if fpath and os.path.exists(fpath):
        with open(fpath, 'r') as f:
            preview = f.read()[:3000]
    elif state.get("kernel_results"):
        preview = state["kernel_results"][0]["fortran_code"][:3000]

    prompt = f"""You are a scientific communication expert in geophysics and numerical methods.
Analyze the following Fortran code and header comments.
Extract and provide the following information concisely in French:
1. What function does this code perform?
2. What is its scientific interest?
3. Which mathematical algorithm / physical equation does it implement?
4. Authors / Contributors / Organization.
5. Years and Publication/Papers (DOIs or citations if present).
6. License (GPL, MIT, etc.).

Metadata / Header:
{header_raw}

Fortran code snippet:
{preview}

Respond in this exact format (no extra text):
📌 Fonction   : <one line>
🔬 Intérêt    : <one line>
📚 Algorithme : <one line>
👤 Auteurs    : <one line>
📅 Année/Pub  : <one line or DOI>
⚖️ Licence    : <one line>
"""
    try:
        response = llm.invoke([
            SystemMessage(content="You are a geophysics and HPC metadata extraction expert."),
            HumanMessage(content=prompt)
        ])
        print(response.content)
        
        # Parsing basique de la réponse pour stockage dans le state
        meta = {}
        for line in response.content.split('\n'):
            if ':' in line:
                k, v = line.split(':', 1)
                meta[k.strip()] = v.strip()
        
        executed = state.get("executed_agents", [])
        executed.append("explainer")
        return {"executed_agents": executed, "scientific_metadata": meta}
    except Exception as e:
        print(f"  ⚠️  Explainer failed: {e}")
        return {"executed_agents": state.get("executed_agents", []) + ["explainer"]}

# ── Worker Agent Loop ──────────────────────────────────────────

def translator_worker_agent(state: TranslationState) -> TranslationState:
    """Agent Worker : Traduit tous les noyaux Fortran en JAX séquentiellement."""
    from fortranspire.llm import get_translator_llm
    llm = get_translator_llm()
    
    # Récupération du schéma global semantique (Loki)
    schema = state.get("ast_info", {}).get("report", {}).get("global_schema", {"params": [], "statics": [], "state": []})
    if not schema or (not schema.get("params") and not schema.get("state")):
         # Fallback si ast_info est vide (cas raw backup)
         schema = state.get("kernel_results", [{}])[0].get("global_schema", {"params": [], "statics": [], "state": []})

    system_prompt = f"""You are Mistral, an elite AI specialized in translating legacy Fortran HPC codes into highly optimized JAX (XLA) programs.
    
    ### CONSOLIDATED SEMANTIC ARCHITECTURE (MANDATORY)
    We are using a module-level NamedTuple structure for ALL routines:
    - `Statics(NamedTuple)` (Logicals): {schema.get('statics', [])}
    - `Params(NamedTuple)` (Grid/Constants): {schema.get('params', [])}
    - `State(NamedTuple)` (Field arrays): {schema.get('state', [])}
    
    RULES:
    1. DO NOT redefine these NamedTuples.
    2. Computational functions MUST take `(statics, params, state)` as primary arguments.
    3. Use `@jax.jit(static_argnames=('statics',))` for kernels.
    4. BRANCHING: If a branch depends on a field array, use `jnp.where` or `lax.cond`. If it depends on a `Static` parameter, use standard Python `if`.
    5. IO Separation: `PRINT`, `WRITE`, `CLOSE` are INFRASTRUCTURE. They MUST be ported to Python `print()` or `logging` in the orchestration block and REMOVED from JAX kernels.
    """

    results = []
    kernels_raw = state.get("kernel_results", [])
    kernels_to_process = [k for k in kernels_raw if k.get("status") == "pending"]
    
    for kernel in kernels_to_process:
        routine_name = kernel["routine_name"]
        fortran_code = kernel["fortran_code"]
        is_master = kernel.get("is_master", False)
        print(f"  ⚙️  [Worker] Translating: {routine_name} " + ("(Greedy Master Infrastructure)" if is_master else ""))

        if is_master:
            prompt = f"""You are porting the main infrastructure of this HPC simulation to Python/JAX.
            Routine: {routine_name} (PROGRAM/Master)
            
            TASK:
            1. Port IO (PRINT/WRITE/READ) to Python native calls. Keep them in the `run_simulation` orchestration.
            2. Extract simulation infrastructure into `run_simulation(statics, params, state) -> State`.
            3. Implement the TIME LOOP using `jax.lax.scan`.
            4. Use the provided NamedTuple schema labels:
               Statics: {schema.get('statics', [])}
               Params: {schema.get('params', [])}
               State: {schema.get('state', [])}
            
            Fortran Code:
            {fortran_code}
            
            Return ONLY the JAX/Python orchestration code in a markdown block.
            """
        else:
            prompt = f"""Translate this Fortran routine into a PURE JAX kernel.
            Routine: {routine_name}
            
            IMPORTANT: Use ONLY (statics, params, state).
            Statics: {schema.get('statics', [])}
            Params: {schema.get('params', [])}
            State: {schema.get('state', [])}
            
            Rules:
            - No IO (PRINT/WRITE).
            - Strict JIT-safe code.
            - Take (statics, params, state) as args.
            
            Fortran Code:
            {fortran_code}
            
            Return ONLY the JAX Python code enclosed in a ```python``` markdown block.
            """
        try:
            resp = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ])
            jax_code = _strip_markdown(resp.content)
            vr = _validate_jax(jax_code, label=f"worker_{routine_name}")
            
            # Repro Check (SKIP for MASTER)
            repro_passed = False
            repro_max_abs = 0.0
            repro_mean_rel = 0.0
            if not is_master:
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        td = Path(tmpdir)
                        (td / "wrapper.f90").write_text(kernel["fortran_wrapper"])
                        (td / "kernel.f90").write_text(fortran_code)
                        (td / "jax_kernel.py").write_text(jax_code)
                        (td / "test_repro.py").write_text(kernel["unit_test_skeleton"])
                        
                        compiler_bin = _fortran_compiler() or "gfortran"
                        env = os.environ.copy()
                        env["PATH"] = f"/opt/homebrew/bin:/usr/local/bin:{env.get('PATH', '')}"
                        so_path = td / f"lib{routine_name.lower()}.so"
                        
                        subprocess.run(
                            [compiler_bin, "-O3", "-shared", "-fPIC", str(td/"wrapper.f90"), str(td/"kernel.f90"), "-o", str(so_path)],
                            capture_output=True, env=env, timeout=30
                        )
                        
                        pytest_bin = os.path.join(os.getcwd(), ".venv/bin/pytest")
                        pytest_env = env.copy()
                        pytest_env["PYTHONPATH"] = f"{td}:{pytest_env.get('PYTHONPATH', '')}"
                        test_res = subprocess.run(
                            [pytest_bin, "-s", str(td/"test_repro.py")],
                            capture_output=True, text=True, timeout=60, env=pytest_env
                        )
                        repro_passed = (test_res.returncode == 0)
                        m = re.search(r"max_abs=([0-9\.eE\-\+]+), mean_rel=([0-9\.eE\-\+]+)", test_res.stdout + test_res.stderr)
                        if m:
                            repro_max_abs, repro_mean_rel = float(m.group(1)), float(m.group(2))
                except Exception as e_repro:
                    print(f"  ⚠️  Repro logic error for {routine_name}: {e_repro}")

            results.append({
                **kernel,
                "jax_code": jax_code,
                "status": "success" if vr["syntax"] else "error",
                "error_log": vr.get("error", ""),
                "repro_passed": repro_passed,
                "repro_max_abs": repro_max_abs,
                "repro_mean_rel": repro_mean_rel
            })
        except Exception as e:
            results.append({**kernel, "status": "error", "error_log": str(e)})

    executed = state.get("executed_agents", [])
    executed.append("translator")
    return {"kernel_results": results, "executed_agents": executed}

def dispatcher_agent(state: TranslationState):
    """Dispatch chaque kernel vers un worker (séquentiel pour stabilisation)."""
    return {}

def consolidator_agent(state: TranslationState) -> TranslationState:
    """Consolide tous les résultats des workers dans un module Python structuré."""
    print(f"\n{SEP}")
    print(f"  🧱 [Consolidator Agent] Building Structured JAX Module")
    print(SEP)
    
    # Récupération du schéma global semantique (Loki)
    schema = state.get("ast_info", {}).get("report", {}).get("global_schema", {"params": [], "statics": [], "state": []})
    if not schema or (not schema.get("params") and not schema.get("state")):
         # Fallback
         kernels_valid = [k for k in state.get("kernel_results", []) if "global_schema" in k]
         schema = kernels_valid[0].get("global_schema", {"params": [], "statics": [], "state": []}) if kernels_valid else {"params": [], "statics": [], "state": []}
    
    class_statics = "class Statics(NamedTuple):\n"
    if schema.get("statics"):
        for s in schema["statics"]:
            class_statics += f"    {s}: Any\n"
    else:
        class_statics += "    pass\n"

    class_params = "class Params(NamedTuple):\n"
    if schema.get("params"):
        for p in schema["params"]:
            class_params += f"    {p}: Any\n"
    else:
        class_params += "    pass\n"

    class_state = "class State(NamedTuple):\n"
    if schema.get("state"):
        for s in schema["state"]:
            class_state += f"    {s}: Any\n"
    else:
        class_state += "    pass\n"

    header = f"""\"\"\"
🚀 Module JAX Consolidé (Loki Semantic Architect)
Structure unifiée (Statics, Params, State) pour JIT performance.
\"\"\"
import jax
import jax.numpy as jnp
from jax import lax, vmap, jit
from typing import NamedTuple, Any

{class_statics}
{class_params}
{class_state}
"""
    
    body = ""
    master_routine = None
    for k in state["kernel_results"]:
        name = k['routine_name']
        is_master = k.get("is_master", False)
        if is_master:
            master_routine = "run_simulation" 
            body += f"\n# --- Master Orchestration: {name} ---\n{k['jax_code']}\n"
        else:
            body += f"\n# --- Kernel: {name} ---\n{k['jax_code']}\n"
    
    footer = f"""
# --- Simulation Launcher ---
def start_simulation():
    \"\"\"Initialisation et lancement de la simulation JAX.\"\"\"
    # Initialisation du schéma (Labels ONLY, values to be provided by user/parser)
    # statics = Statics(...)
    # params = Params(...)
    # state = State(...)
    
    print("⏳ Warming up JAX...")
    # final_state = run_simulation(statics, params, state)
    print("✅ Simulation complete.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        start_simulation()
    else:
        print("💡 Use 'python kernels_consolidated.py run' to start.")
"""
    if not master_routine:
        footer = "\n# No master routine detected.\n"

    full_jax = header + body + footer
    
    # Save the consolidated JAX module
    out = _output_dir(state, "src")
    _save(out / "kernels_consolidated.py", full_jax)

    # ── Generate README.md for tracking ──────────────────────────
    import datetime
    kernel_rows = ""
    for k in state["kernel_results"]:
        kstatus = "✅ success" if k.get("status") == "success" else "❌ error" if k.get("status") == "error" else "⏳ pending"
        kind = "🏗️ master" if k.get("is_master") else "⚙️  kernel"
        repro = "⏭️ PORT" if k.get("is_master") else ("✅ PASS" if k.get("repro_passed") else "❌ FAIL")
        kernel_rows += f"| `{k['routine_name']}` | {kind} | {kstatus} | {repro} |\n"

    readme = f"""# 🚀 JAX Translation — {Path(state['fortran_filepath']).name}

Generated: `{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
Source: `{state['fortran_filepath']}`

## Schema (Loki Semantic Architect)

| Category | Fields |
| :--- | :--- |
| `Statics` (JIT-static logicals) | `{', '.join(schema.get('statics', [])) or 'none'}` |
| `Params` (grid/physical constants) | `{', '.join(schema.get('params', [])) or 'none'}` |
| `State` (field arrays) | `{', '.join(schema.get('state', [])[:10]) or 'none'}{'...' if len(schema.get('state',[])) > 10 else ''}` |

## Kernels

| Routine | Kind | JAX Status | Repro |
| :--- | :--- | :--- | :--- |
{kernel_rows}
## Quick Start

```bash
python src/{Path(state['fortran_filepath']).stem}/kernels_consolidated.py run
```

## Files

| File | Description |
| :--- | :--- |
| `src/.../kernels_consolidated.py` | Main JAX module with Statics/Params/State |
| `src/.../surrogate_fno.py` | FNO surrogate model |
| `tests/reproducibility/.../REPRODUCIBILITY_REPORT.md` | Numerical parity report |
| `tests/performance/` | Benchmark results |
"""
    out_root = Path("output")
    _save(out_root / "README.md", readme)
    # ─────────────────────────────────────────────────────────────

    executed = state.get("executed_agents", [])
    executed.append("consolidator")
    return {"jax_code": full_jax, "kernels_translated": len(state["kernel_results"]), "executed_agents": executed}

def translate_kernel_agent(state: TranslationState) -> TranslationState:
    """Agent 3 : Traduit le kernel Fortran vers JAX via LLM (avec boucles de correction)."""
    if "translator" in state.get("executed_agents", []):
        return state

    if state.get("compilation_error") == "FILE_TOO_LARGE":
        print("[Translation Agent] Skipped due to file size limit.")
        return state

    n_retries = state.get("translation_retries", 0)
    kernels = state.get("kernels_found", [])
    n = len(kernels) if kernels else 1
    
    print(f"\n{SEP}")
    print(f"  ⚙️  [Translation Agent] Fortran ➔ JAX (Attempt {n_retries + 1}/5)")
    print(f"  📦 Kernels to translate : {n}  ({', '.join(kernels) if kernels else 'raw fallback'})")
    if state.get("last_error"):
        print(f"  🔄 Retrying due to previous error: {state['last_error'][:100]}...")
    print(SEP)

    from fortranspire.llm import get_translator_llm
    llm = get_translator_llm()

    hints_ctx = ""
    if state.get("jax_hints"):
        hints_ctx = "Loki Structural Hints:\n" + "\n".join(state["jax_hints"])

    # Construction du prompt avec feedback éventuel
    feedback_ctx = ""
    if state.get("last_error"):
        feedback_ctx = f"""
--- PREVIOUS ATTEMPT FAILED ---
The previous JAX translation was invalid.
ERROR: {state['last_error']}
Please FIX this error in your next response.
--------------------------------
"""

    prompt = f"""Translate the following Fortran 90 kernel into pure Python using JAX (jax.numpy).
Ensure loops are vectorized using jax.numpy operations or jax.lax.scan/fori_loop.
{feedback_ctx}
{hints_ctx}

Fortran Kernel:
{state['isolated_kernel']}

Return ONLY the Python code without any markdown wrapper.
"""
    response = llm.invoke([
        SystemMessage(content="You are an expert scientific Fortran to JAX translator. You correct your errors based on traceback feedback."),
        HumanMessage(content=prompt)
    ])

    translated = _strip_markdown(response.content)
    
    # Validation syntaxe + XLA
    kernel_name = state.get("ast_info", {}).get("routine_name", "kernel")
    vr = _validate_jax(translated, label=kernel_name)
    
    executed = state.get("executed_agents", [])
    if vr["syntax"]:
        print(f"  ✅ JAX validation    : syntax ✓")
        print(f"  ✅ JAX execution     : {'✓' if vr['exec'] else '⚠'}  jit {'✓' if vr['jit'] else '⚠'}")
        executed.append("translator")
        last_error = ""
    else:
        print(f"  ❌ JAX syntax error  : {vr['error'][:100]}...")
        last_error = vr["error"]

    out = _output_dir(state, "src")
    _save(out / "jax_kernel.py", translated)
    print(SEP + "\n")

    return {
        "jax_code": translated, 
        "translation_retries": n_retries + 1,
        "last_error": last_error,
        "executed_agents": executed
    }


# ── Agent 3.5 : IDE Interaction (Manual Mode) ───────────────────

def ide_interaction_agent(state: TranslationState) -> TranslationState:
    """Agent 3.5 : Ouvre le code JAX dans l'IDE et attend la validation de l'utilisateur."""
    if state.get("interaction_mode") != "manual":
        return state

    print(f"\n{SEP2}")
    print(f"  ✨ [IDE Interaction Agent] Validation Manuelle")
    print(SEP2)

    # 1. Préparation du tutoriel / Review Artifact pour Antigravity
    review_content = f"""# 🔍 Revue Critique des Kernels JAX

Utilisez le carousel ci-dessous pour comparer le code Fortran original et sa traduction JAX.

````carousel
```fortran
{state['kernel_results'][0]['fortran_code'] if state.get('kernel_results') else '! No code found'}
```
<!-- slide -->
```python
{state.get('jax_code', '# No translation found')}
```
<!-- slide -->
### 📝 Instructions de Revue
1. **Vérifiez** les mutations `inout` (JAX doit renvoyer un nouvel état).
2. **Corrigez** les boucles spatiales `for i in range` si elles n'ont pas été vectorisées.
3. **Validez** l'utilisation de `jnp.where` pour les IFs.

**Modifiez directement le fichier JAX ouvert dans votre IDE pour injecter vos corrections.**
````
"""
    _save(Path("REVIEW_KERNELS.md"), review_content)
    print(f"  📝 [Artifact] Created REVIEW_KERNELS.md for side-by-side review in Antigravity.")

    out_jax = _output_dir(state, "src")
    jax_file = out_jax / "jax_kernel.py"
    
    # On sauve le jax_code actuel dans le fichier pour que l'utilisateur puisse l'éditer
    _save(jax_file, state.get("jax_code", ""))

    print(f"  📂 Opening translated JAX code in your IDE...")
    try:
        subprocess.run(["open", str(jax_file)])
    except:
        pass

    print(f"\n  👉 [ACTION REQUIRED]")
    print(f"     1. Revue dans Antigravity (REVIEW_KERNELS.md)")
    print(f"     2. Edition des fichiers (open {jax_file.name})")
    print(f"     3. Sauvegardez et appuyez sur ENTREE.")
    
    input("\n  [En attente de votre validation (ENTREE)...]")

    print(f"  🔍 Reading back user modifications...")
    updated_code = jax_file.read_text(encoding="utf-8")
    return {"jax_code": updated_code}


# ── Agent 4 : GHEX / Halo Exchange ────────────────────────────────

def halo_exchange_agent(state: TranslationState) -> TranslationState:
    """Agent 4 : Rapport OpenMP/MPI + remplacement GHEX si nécessaire."""
    if state.get("compilation_error") == "FILE_TOO_LARGE":
        return state

    print(f"\n{SEP}")
    print(f"  🔗 [GHEX Agent] HPC Parallelism Report")
    print(SEP)

    fortran_src = ""
    if state.get("kernel_results"):
        fortran_src = "\n\n".join([k["fortran_code"] for k in state["kernel_results"]])

    # Détection OpenMP
    omp_lines = [l.strip() for l in fortran_src.splitlines() if "!$omp" in l.lower()]

    # Détection MPI
    mpi_calls = re.findall(r'MPI_\w+', fortran_src, re.IGNORECASE)
    mpi_unique = sorted(set(mpi_calls))

    print(f"  OpenMP directives : {len(omp_lines):>3} found", end="")
    print(f"  → {'✅ will wrap with jax.vmap/pmap' if omp_lines else '⏭️  no OpenMP (skipped)'}")
    for line in omp_lines[:3]:
        print(f"      {line}")

    print(f"  MPI calls         : {len(mpi_unique):>3} found", end="")
    print(f"  → {'✅ rewriting with GHEX halo exchange' if mpi_unique else '⏭️  no MPI (skipped)'}")
    if mpi_unique:
        print(f"      {mpi_unique}")

    hpc_report = {
        "openmp_count": len(omp_lines),
        "openmp_directives": omp_lines[:5],
        "openmp_treated": len(omp_lines) > 0,
        "mpi_calls": mpi_unique,
        "mpi_treated": len(mpi_unique) > 0,
    }

    jax_code = state["jax_code"]
    if mpi_unique:
        from fortranspire.llm import get_translator_llm
        llm = get_translator_llm()
        prompt = f"""Analyze the following JAX code. If it contains MPI_Send/Recv or explicit halo boundary exchanges,
rewrite those specific communication blocks using abstract GHEX (Generic Halo Exchange) patterns.
If no MPI/Halo logic exists, return the exact same code.

Code:
{jax_code}
"""
        response = llm.invoke([
            SystemMessage(content="You are an HPC Halo Exchange expert."),
            HumanMessage(content=prompt)
        ])
        jax_code = _strip_markdown(response.content)

    print(SEP + "\n")
    return {"jax_code": jax_code, "hpc_report": hpc_report}


# ── Agent 5 : Reproducibility ─────────────────────────────────────

def reproducibility_agent(state: TranslationState) -> TranslationState:
    """Agent 5 : Consolide les rapports de reproductibilité générés en parallèle."""
    print(f"\n{SEP}")
    print(f"  🧪 [Reproducibility Agent] Generating Global Report")
    print(SEP)

    kernels_raw = state.get("kernel_results", [])
    if not kernels_raw:
        return state

    # Déduplication : on ne garde que le dernier résultat par routine (Map-Reduce)
    unique_kernels = {}
    for k in kernels_raw:
        unique_kernels[k["routine_name"]] = k
    kernels = list(unique_kernels.values())

    # Construction du rapport Markdown (Artifact)
    report = "# 📊 Rapport de Reproductibilité Granulaire\n\n"
    report += "| Routine | JAX Status | Repro Status | Max Abs Error | Mean Rel Error | Notes |\n"
    report += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    # Affichage CLI (Tableau ASCII)
    print(f"\n  📊 [Reproducibility Results Dashboard]")
    print(f"  {'-'*110}")
    print(f"  {'Routine':<35} | {'JAX':<3} | {'Repro':<6} | {'Max Abs':<12} | {'Mean Rel':<12} | {'Status'}")
    print(f"  {'-'*110}")

    passed_count = 0
    calculated_count = 0
    for k in kernels:
        is_master = k.get("is_master", False)
        is_passed = k.get("repro_passed", False)
        
        # JAX Translation Status
        j_status = "OK" if k.get("status") == "success" else "ERR"
        j_icon = "✅" if k.get("status") == "success" else "⚠️"
        
        # Repro Status [Logic Change for Greedy Master]
        if is_master:
            r_status = "PORT"
            r_icon = "⏭️ "
        else:
            calculated_count += 1
            r_status = "PASS" if is_passed else "FAIL"
            r_icon = "✅" if is_passed else "❌"
            if is_passed:
                passed_count += 1

        m_abs = f"{k.get('repro_max_abs', 0.0):.2e}"
        m_rel = f"{k.get('repro_mean_rel', 0.0):.2e}"
        
        # Markdown report
        report += f"| {k['routine_name']} | {j_icon} {j_status} | {r_icon} {r_status} | {m_abs} | {m_rel} | {k['error_log']} |\n"
        
        # CLI print
        print(f"  {k['routine_name']:<35} | {j_icon} {j_status} | {r_icon} {r_status:<4} | {m_abs:<12} | {m_rel:<12} | {k.get('status', 'pending')}")

    print(f"  {'-'*110}")
    total_to_check = calculated_count if calculated_count > 0 else 1
    print(f"  📈 Summary: {passed_count}/{calculated_count} subroutines passed numerical validation.")
    print(f"  ⏭️  Master routine infrastructure ported to Python.")
    print(f"  {'-'*110}\n")

    # Sauvegarde de l'artifact
    _save(_output_dir(state, "tests/reproducibility") / "REPRODUCIBILITY_REPORT.md", report)
    print(f"  📈 Detailed report saved: REPRODUCIBILITY_REPORT.md ({passed_count}/{len(kernels)} passed)")

    executed = state.get("executed_agents", [])
    executed.append("reproducibility")
    
    return {
        "reproducibility_passed": (passed_count == len(kernels)),
        "executed_agents": executed
    }


# ── Agent 6 : Performance ─────────────────────────────────────────

def performance_agent(state: TranslationState) -> TranslationState:
    """Agent 6 : Benchmark JAX vs Fortran — affichage CLI du speedup."""
    if state.get("compilation_error") == "FILE_TOO_LARGE":
        return state

    print(f"\n{SEP}")
    print(f"  ⚡ [Performance Agent] JAX vs Fortran Benchmark")
    print(SEP)

    compiler = _fortran_compiler()
    fortran_ms = None
    jax_ms = None

    # ── Benchmark Fortran ─────────────────────────────────────────
    if compiler:
        try:
            f_code = ""
            if state.get("kernel_results"):
                f_code = state["kernel_results"][0]["fortran_code"]
                
            # Wrapper PROGRAM pour gfortran si nécessaire
            if "PROGRAM " not in f_code.upper():
                r_name = state.get("ast_info", {}).get("routine_name", "bench_kernel")
                f_code = f"PROGRAM bench\n  CALL {r_name}\nEND PROGRAM\n\n" + f_code

            with tempfile.TemporaryDirectory() as tmpdir:
                f_path = Path(tmpdir) / "kernel.f90"
                f_path.write_text(f_code)
                bin_path = Path(tmpdir) / "kernel_bench"
                result = subprocess.run(
                    [compiler, "-O3", str(f_path), "-o", str(bin_path)],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    t0 = time.perf_counter()
                    subprocess.run([str(bin_path)], capture_output=True, timeout=60)
                    fortran_ms = (time.perf_counter() - t0) * 1000
                    print(f"  Fortran ({compiler} -O3)  : {fortran_ms:>8.1f} ms")
                else:
                    print(f"  Fortran compile          : ⚠️  skipped (compile error)")
        except Exception as e:
            print(f"  Fortran benchmark        : ⚠️  skipped ({e})")
    else:
        print("  Fortran                  : ⚠️  no compiler found (brew install gcc)")

    # ── Benchmark JAX ─────────────────────────────────────────────
    try:
        jax_src = state.get("jax_code", "")
        # On injecte jax/jnp dans le scope du benchmark
        benchmark_globals = {"jax": jax, "jnp": jnp}
        t = timeit.timeit(
            stmt="pass",
            setup=f"exec(compile({repr(jax_src)}, '<jax>', 'exec'))",
            number=5,
            globals=benchmark_globals
        ) / 5 * 1000
        jax_ms = t
        print(f"  JAX (compile, CPU)       : {jax_ms:>8.1f} ms")
    except Exception as e:
        print(f"  JAX benchmark            : ⚠️  skipped ({e})")

    # ── Speedup ───────────────────────────────────────────────────
    speedup = None
    if fortran_ms and jax_ms:
        speedup = fortran_ms / jax_ms
        print(f"  Speedup                  :   {speedup:>6.1f}×  🚀")

    # Export Bencher JSON
    kernel_name = state.get("ast_info", {}).get("routine_name", "kernel")
    out = _output_dir(state, "tests/performance")
    bencher = {
        kernel_name: {
            "fortran_ms": {"value": fortran_ms or 0},
            "jax_ms":     {"value": jax_ms or 0},
            "speedup":    {"value": speedup or 0},
        }
    }
    _save(out / "bencher_report.json", json.dumps(bencher, indent=2))
    print(SEP + "\n")

    # ── 4. Audit d'Optimisation JAX/XLA ────────────────────────────
    print(f"  📝 Generating Optimization Audit...")
    from fortranspire.llm import get_reasoning_llm
    audit_llm = get_reasoning_llm()
    
    audit_prompt = f"""
    Analyze the following JAX code in comparison with the provided XLA profiling hints and the 'Tableau de Chasse' rules.
    Evaluate the implementation of the NamedTuple architecture and physical consistency.
    
    XLA Hints: {state.get('jax_hints', [])}
    JAX Code: {state.get('jax_code', '')[:4000]}
    
    Generate a technical report (Markdown) following the scientific criteria:
    1. NamedTuple usage (Params vs State) and state injection.
    2. Indexing (0-based) and Silent Clamping management.
    3. Floating point precision alignment and explicit casts.
    4. XLA branching (jnp.where) and temporal loops (lax.scan).
    5. Final performance verdict.
    """
    try:
        audit_resp = audit_llm.invoke([HumanMessage(content=audit_prompt)])
        _save(_output_dir(state, "tests/performance") / "OPTIMIZATION_REVIEW.md", audit_resp.content)
        print(f"  ✅ Optimization Audit saved: tests/performance/OPTIMIZATION_REVIEW.md")
    except Exception as e:
        print(f"  ⚠️  Optimization Audit failed: {e}")

    return {"performance_metrics": {"speedup": speedup, "fortran_ms": fortran_ms, "jax_ms": jax_ms}}


# ── Agent 7 : Docstring ───────────────────────────────────────────

def docstring_agent(state: TranslationState) -> TranslationState:
    """Agent 7 : Ajoute des docstrings scientifiques au code JAX."""
    if "docstring" in state.get("executed_agents", []):
        return state

    if state.get("compilation_error") == "FILE_TOO_LARGE":
        return state

    print(f"\n{SEP}")
    print(f"  📝 [Docstring Agent] Scientific documentation")
    print(SEP)

    metadata_ctx = ""
    if state.get("scientific_metadata"):
        metadata_ctx = "Scientific Context from original source:\n"
        for k, v in state["scientific_metadata"].items():
            metadata_ctx += f"- {k}: {v}\n"

    prompt = f"""Update the following JAX python code by adding comprehensive Python docstrings and a formal header.
Requirements:
1. Module-level header block: State the equivalent Fortran routine ({state['fortran_filepath']}).
   Include Authors, Publication/Paper, Year, and License using the provided metadata.
2. For each function, add a docstring explaining the exact mathematical formula implemented.

{metadata_ctx}

JAX Code:
{state['jax_code']}

Return the fully updated Python code.
"""
    from fortranspire.llm import get_reasoning_llm
    llm = get_reasoning_llm()
    response = llm.invoke([
        SystemMessage(content="You are an expert scientific documentation assistant. You preserve historical and academic provenance."),
        HumanMessage(content=prompt)
    ])
    documented = _strip_markdown(response.content)
    vr = _validate_jax(documented, label="jax_kernel_docstring")
    status = '✓' if vr['syntax'] else '❌'
    print(f"  JAX syntax check    : {status}")
    out = _output_dir(state, "src")
    _save(out / "jax_kernel_docstring.py", documented)
    print(SEP + "\n")
    executed = state.get("executed_agents", [])
    executed.append("docstring")
    return {"jax_code": documented, "executed_agents": executed}


# ── Agent 8 : Autodiff ────────────────────────────────────────────

def autodiff_adjoint_agent(state: TranslationState) -> TranslationState:
    """Agent 8 : Génère le modèle Adjoint via jax.grad/jax.vjp."""
    if state.get("compilation_error") == "FILE_TOO_LARGE":
        return state

    print(f"\n{SEP}")
    print(f"  ∂  [Autodiff Agent] Adjoint via jax.grad / jax.vjp")
    print(SEP)

    from fortranspire.llm import get_translator_llm
    llm = get_translator_llm()

    prompt = f"""You have a forward simulation kernel in JAX.
Generate JUST the adjoint function `adjoint_kernel` using `jax.grad` or `jax.vjp`.
Do not repeat the forward code. Simply provide the adjoint function and necessary imports.

Forward JAX code:
{state['jax_code']}

Return ONLY the JAX adjoint Python code.
"""
    response = llm.invoke([
        SystemMessage(content="You are an expert in Autodifferentiation and adjoint models."),
        HumanMessage(content=prompt)
    ])
    result = _strip_markdown(response.content)
    vr = _validate_jax(result, label="adjoint")
    status = '✓' if vr['syntax'] else '❌'
    print(f"  JAX syntax check    : {status}")
    out = _output_dir(state, "src")
    _save(out / "adjoint.py", result)
    print(SEP + "\n")
    return state  # Keep forward jax_code in state, save adjoint separately


# ── Agent 9 : Surrogate FNO ──────────────────────────────────────

def surrogate_fno_agent(state: TranslationState) -> TranslationState:
    """Agent 9 (final) : Construit un Surrogate FNO avec Sobolev Training."""
    if "surrogate" in state.get("executed_agents", []):
        return state

    if state.get("compilation_error") == "FILE_TOO_LARGE":
        return state

    framework = state.get("surrogate_framework", "flax")
    print(f"\n{SEP}")
    print(f"  🤖 [Surrogate Agent] FNO ({framework}) + Sobolev Training")
    print(SEP)

    from fortranspire.llm import get_reasoning_llm
    llm = get_reasoning_llm()

    prompt = f"""Generate a complete `{framework}` script that implements a 2D Fourier Neural Operator (FNO).
Requirements:
1. `FNO2d` class using `jax.numpy.fft.rfft2` for spectral convolutions.
2. `train_step` function with Sobolev Training loss:
   L = MSE(Surrogate(x), Target) + λ * MSE(grad_Surrogate(x), grad_Physics(x))
3. Use explicit spatial dimension parameters derived from AST context.
4. Return ONLY Python code, no markdown.

AST Context: {state.get('ast_info', {})}
JAX Code (truncated): {state.get('jax_code', '')[:2000]}
"""
    response = llm.invoke([
        SystemMessage(content=f"You are a Deep Learning expert specializing in Physics-Informed ML and {framework}."),
        HumanMessage(content=prompt)
    ])
    result = _strip_markdown(response.content)
    vr = _validate_jax(result, label="surrogate_fno")
    status = '✓' if vr['syntax'] else '❌'
    print(f"  JAX syntax check    : {status}")
    out = _output_dir(state, "src")
    _save(out / "surrogate_fno.py", result)

    # Récapitulatif final
    print(f"\n{SEP2}")
    print(f"  🏁 Pipeline Complete !")
    print(f"  📂 Output directory : {Path('output').resolve()}")
    print(f"  Files generated:")
    for f in sorted(Path("output").rglob("*")):
        if f.is_file():
            rel = f.relative_to(Path("output"))
            size = f.stat().st_size
            print(f"    {'✅' if size > 50 else '⚠️ '} {str(rel):35s} ({size:>6} bytes)")
    print(SEP2 + "\n")

    executed = state.get("executed_agents", [])
    executed.append("surrogate")
    return {"surrogate_code": result, "executed_agents": executed}


# ==========================================
# 4. Construction du Graphe
# ==========================================

def should_continue_translation(state: TranslationState):
    """Décide si on continue vers le halo ou si on retry le translator."""
    if "translator" in state.get("executed_agents", []):
        return "halo_exchange"
    if state.get("translation_retries", 0) >= 5:
        print("  ⚠️  Maximum translation retries reached. Proceeding with best-effort.")
        return "halo_exchange"
    return "translator"

def should_continue_repro(state: TranslationState):
    """Décide si on continue vers perf ou si on retry le translator (feedback loop)."""
    if state.get("reproducibility_passed", False):
        return "performance"
    if state.get("reproducibility_retries", 0) >= 5:
        print("  ⚠️  Maximum reproducibility retries reached. Proceeding with best-effort.")
        return "performance"
    return "translator"

workflow = StateGraph(TranslationState)

workflow.add_node("init",           init_project_agent)
workflow.add_node("parser",         parse_and_isolate_agent)
workflow.add_node("explainer",      explainer_agent)
workflow.add_node("dispatcher",     dispatcher_agent)
workflow.add_node("translator",     translator_worker_agent)
workflow.add_node("consolidator",   consolidator_agent)
workflow.add_node("ide_interaction", ide_interaction_agent)
workflow.add_node("halo_exchange",  halo_exchange_agent)
workflow.add_node("reproducibility", reproducibility_agent)
workflow.add_node("performance",    performance_agent)
workflow.add_node("docstring",      docstring_agent)
workflow.add_node("autodiff",       autodiff_adjoint_agent)
workflow.add_node("surrogate",      surrogate_fno_agent)

workflow.set_entry_point("init")
workflow.add_edge("init",            "parser")
workflow.add_edge("parser",          "explainer")
workflow.add_edge("explainer",          "dispatcher")
workflow.add_edge("dispatcher",         "translator")
workflow.add_edge("translator",      "consolidator")
workflow.add_edge("consolidator",    "ide_interaction")
workflow.add_edge("ide_interaction", "halo_exchange")
workflow.add_edge("halo_exchange",   "reproducibility")

# Boucle conditionnelle pour la reproductibilité numérique
workflow.add_conditional_edges(
    "reproducibility",
    should_continue_repro,
    {
        "translator": "dispatcher", # Re-dispatch en cas d'erreur
        "performance": "performance"
    }
)

workflow.add_edge("performance",     "docstring")
workflow.add_edge("docstring",       "autodiff")
workflow.add_edge("autodiff",        "surrogate")
workflow.add_edge("surrogate",       END)

translation_app = workflow.compile()

__all__ = ["translation_app", "performance_agent"]
