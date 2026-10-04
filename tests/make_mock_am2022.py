"""Build a mock FatigueData-AM2022 JSON (documented schema) from the synthetic generator, to test the loader."""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from amf import synth

NAMES = {"Ti64": "Ti-6Al-4V", "AlSi10Mg": "AlSi10Mg", "316L": "316L", "IN718": "IN718"}


def main(out, n=90, seed=3):
    st = synth.generate(n, seed=seed)
    rng = np.random.default_rng(seed)
    arts, i = [], 0
    while i < len(st):
        k = int(rng.integers(1, 4))
        dsets = []
        for s in st[i:i + k]:
            proc = [{"type": "HIP", "temperature": 920, "time": 2, "pressure": 100}] if s["hip"] else (
                [{"type": "HT", "temperature": 650, "time": 2}] if s["ht"] else [{"type": "NHT"}])
            proc.append({"type": "SURF", "method": "machined" if s["mach"] else "as-built"})
            dsets.append({"fatigue": {"fat_data": np.column_stack([s["N"], s["sa"], 1 - s["d"]]).tolist(),
                                      "fdata_type": "S-N", "extract_method": "color and shape"},
                          "materials": {"mat_name": NAMES[s["alloy"]]},
                          "am": {"am_type": "E-PBF" if s["ebm"] else "L-PBF",
                                 "am_para": {"direction": 90.0 if s["vert"] else 0.0, "layer_thickness": 30}},
                          "processing": {"proc_para": proc},
                          "testing": {"fat_type": "uniaxial", "fat_temp": 25, "fat_env": "air", "fat_r": s["R_study"],
                                      "frequency": 20, "spec_kt": 1, "load_ctrl": "force"},
                          "static": {"tensile_strength": [] if s["uts_reported"] is None else s["uts_reported"]}})
        arts.append({"metadata": {"doi": f"10.0000/mock.{len(arts)}"}, "scientific_data": {"fatigue_datasets": dsets}})
        i += k
    with open(out, "w") as f:
        json.dump({"articles": arts}, f)
    print("wrote", out, len(arts), "articles", n, "datasets")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mock_am2022.json")
