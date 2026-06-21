
from loki import Sourcefile
from loki.ir import FindNodes
from loki.ir.nodes import Declaration, Loop
import os

def test_loki_params(filepath):
    source = Sourcefile.from_file(filepath)
    print(f"File: {filepath}")
    
    params = []
    logicals = []
    
    # Traverse all routines
    for routine in source.routines:
        print(f"\nRoutine: {routine.name}")
        # Look at declarations
        for decl in FindNodes(Declaration).visit(routine.spec):
            for var in decl.symbols:
                # Check attributes carefully
                is_param = getattr(var.type, 'parameter', False)
                dtype = str(getattr(var.type, 'dtype', 'unknown')).lower()
                
                if is_param:
                    print(f"  Param: {var.name} | Type: {dtype}")
                    params.append(var.name)
                    if 'logical' in dtype:
                        logicals.append(var.name)
                else:
                    # Non-parameter arrays could be State
                    if hasattr(var, 'dimensions') and var.dimensions:
                         pass # potential state

    print("\nSummary:")
    print(f"Params: {sorted(list(set(params)))}")
    print(f"Logicals (static): {sorted(list(set(logicals)))}")

if __name__ == "__main__":
    test_loki_params("/Users/loicmaurin/PycharmProjects/seismic_cpml/seismic_CPML_2D_isotropic_second_order.f90")
