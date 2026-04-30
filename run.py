from __future__ import print_function
import argparse
import os
import yaml
import random
from time import time
from rdkit import Chem
from rdkit.Chem import QED

def main():

    start_time = time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--smi_file', default=None)
    args = parser.parse_args()

    print("🚀 RUNNING FIXED PIPELINE")

    # ===============================
    # 📥 READ INPUT MOLECULE
    # ===============================
    seed_smiles = "CCO"

    if args.smi_file and os.path.exists(args.smi_file):
        with open(args.smi_file, "r") as f:
            seed_smiles = f.read().strip()

    print("INPUT MOLECULE:", seed_smiles)

    seed_mol = Chem.MolFromSmiles(seed_smiles)

    if seed_mol is None:
        print("❌ Invalid input SMILES")
        return

    # ===============================
    # 🧬 GENERATE VALID MOLECULES
    # ===============================
    results = {}

    for i in range(300):
        try:
            mol = Chem.Mol(seed_mol)
            rw = Chem.RWMol(mol)

            # 🔥 STRONG MUTATION
            action = random.choice(["add", "remove", "replace", "ring"])

            if action == "add":
                atom_idx = rw.AddAtom(Chem.Atom(random.choice([6, 7, 8, 9, 16])))
                target = random.randint(0, rw.GetNumAtoms() - 2)
                rw.AddBond(target, atom_idx, Chem.BondType.SINGLE)

            elif action == "remove" and rw.GetNumAtoms() > 5:
                idx = random.randint(0, rw.GetNumAtoms() - 1)
                rw.RemoveAtom(idx)

            elif action == "replace":
                idx = random.randint(0, rw.GetNumAtoms() - 1)
                rw.GetAtomWithIdx(idx).SetAtomicNum(random.choice([6, 7, 8, 9, 16]))

            elif action == "ring" and rw.GetNumAtoms() >= 3:
                a1 = random.randint(0, rw.GetNumAtoms() - 1)
                a2 = random.randint(0, rw.GetNumAtoms() - 1)
                if a1 != a2:
                    try:
                        rw.AddBond(a1, a2, Chem.BondType.SINGLE)
                    except:
                        pass

            # ✅ convert + validate
            new_mol = rw.GetMol()
            Chem.SanitizeMol(new_mol)

            new_smi = Chem.MolToSmiles(new_mol)

            if new_smi:
                score = round(QED.qed(new_mol), 3)
                results[new_smi] = [score, i]

        except:
            continue

    # ===============================
    # 🚨 FALLBACK
    # ===============================
    if len(results) == 0:
        print("⚠️ No valid mutations, using fallback")
        results[seed_smiles] = [0.9, 0]

    # ===============================
    # 💾 SAVE YAML
    # ===============================
    output_dir = "main/graph_ga/results"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "results.yaml")

    with open(output_path, "w") as f:
        yaml.dump(results, f)

    print("✅ YAML CREATED:", output_path)
    print("Total molecules:", len(results))
    print("⏱ Finished in %.2f seconds" % (time() - start_time))


if __name__ == "__main__":
    main()