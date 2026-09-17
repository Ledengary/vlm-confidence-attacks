import importlib

MODULES = [
    "src.analysis.tables_main",
    "src.analysis.tables_setup",
    "src.analysis.tables_mechanisms",
    "src.analysis.tables_longtables",
    "src.analysis.tables_phenomenon",
    "src.analysis.tables_distribution",
    "src.analysis.tables_scope",
    "src.figures.frontier",
]


def main():
    total = 0
    for modname in MODULES:
        mod = importlib.import_module(modname)
        for name in sorted(mod.BUILDERS):
            path = mod.BUILDERS[name]()
            print(name, path)
            total += 1
    print(f"built {total}")


if __name__ == "__main__":
    main()
