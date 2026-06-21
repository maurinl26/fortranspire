
from loki import Sourcefile, FindNodes
import os

# Try to find where nodes are
try:
    from loki.ir.nodes import Declaration
except ImportError:
    try:
        from loki.ir import Declaration
    except ImportError:
        Declaration = None

def inspect_loki(filepath):
    source = Sourcefile.from_file(filepath)
    for routine in source.routines:
        print(f"\n--- Routine: {routine.name} ---")
        # Print all attributes of a variable symbol
        if hasattr(routine, 'spec'):
            for node in routine.spec.body:
                print(f"Node: {type(node)}")
                if hasattr(node, 'symbols'):
                    for var in node.symbols:
                        print(f"  Symbol: {var.name}")
                        print(f"    Type: {var.type}")
                        print(f"    Attrs: {dir(var.type)}")
                        print(f"    Is Param: {getattr(var.type, 'parameter', 'N/A')}")
                        print(f"    DType: {getattr(var.type, 'dtype', 'N/A')}")
        
        # Look for IO in the body
        if hasattr(routine, 'body'):
            for node in routine.body.body:
                # Some nodes might be IO
                print(f"Body Node: {type(node)}")

if __name__ == "__main__":
    inspect_loki("/Users/loicmaurin/PycharmProjects/seismic_cpml/seismic_CPML_2D_isotropic_second_order.f90")
