from __future__ import print_function
import random
import os
import yaml
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, DataStructs, QED, Descriptors

# 🔇 suppress RDKit warnings
rdBase.DisableLog('rdApp.error')


# ===============================
# 🔹 SIMILARITY FUNCTION
# ===============================
def compute_similarity(smi1, smi2):
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)

    if mol1 is None or mol2 is None:
        return 0.0

    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    gen = GetMorganGenerator(radius=2)

    fp1 = gen.GetFingerprint(mol1)
    fp2 = gen.GetFingerprint(mol2)

    return DataStructs.TanimotoSimilarity(fp1, fp2)


# ===============================
# 🔹 MUTATION
# ===============================
def mutate(mol):
    try:
        # occasional aromatic ring
        if random.random() < 0.2:
            return Chem.MolFromSmiles("c1ccccc1")

        rw = Chem.RWMol(mol)

        action = random.choice(["add", "add", "replace", "bond"])

        if action == "add":
            atom = Chem.Atom(random.choice([6, 7, 8]))
            idx = rw.AddAtom(atom)

            if rw.GetNumAtoms() > 1:
                target = random.randint(0, rw.GetNumAtoms() - 2)
                rw.AddBond(target, idx, Chem.BondType.SINGLE)

        elif action == "replace":
            idx = random.randint(0, rw.GetNumAtoms() - 1)
            rw.GetAtomWithIdx(idx).SetAtomicNum(random.choice([6, 7, 8]))

        elif action == "bond":
            if rw.GetNumAtoms() >= 2:
                a1 = random.randint(0, rw.GetNumAtoms() - 1)
                a2 = random.randint(0, rw.GetNumAtoms() - 1)

                if a1 != a2 and rw.GetBondBetweenAtoms(a1, a2) is None:
                    rw.AddBond(a1, a2, Chem.BondType.SINGLE)

        new_mol = rw.GetMol()
        Chem.SanitizeMol(new_mol)
        return new_mol

    except:
        return None


# ===============================
# 🔹 CROSSOVER
# ===============================
def crossover(mol1, mol2):
    try:
        smi1 = Chem.MolToSmiles(mol1)
        smi2 = Chem.MolToSmiles(mol2)

        if len(smi1) < 4 or len(smi2) < 4:
            return None

        cut1 = random.randint(1, len(smi1) - 2)
        cut2 = random.randint(1, len(smi2) - 2)

        child_smi = smi1[:cut1] + smi2[cut2:]

        child = Chem.MolFromSmiles(child_smi)

        if child:
            Chem.SanitizeMol(child)
            return child

    except:
        return None

    return None


# ===============================
# 🔹 FITNESS FUNCTION
# ===============================
def fitness(smi, seed_smiles):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0

    qed_score = QED.qed(mol)
    similarity = compute_similarity(seed_smiles, smi)

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)

    penalty = 0
    if mw > 500:
        penalty += 0.2
    if logp > 5:
        penalty += 0.2

    if smi == seed_smiles:
        return 0.1

    if qed_score < 0.2:
        return 0.05

    score = 0.9 * qed_score + 0.1 * similarity - penalty
    score += random.uniform(0, 0.03)

    return round(max(score, 0), 3)


# ===============================
# 🔹 GENETIC ALGORITHM
# ===============================
def run_ga(seed_smiles, generations=15, population_size=40):

    print("INPUT:", seed_smiles)

    population = [seed_smiles]
    seed_mol = Chem.MolFromSmiles(seed_smiles)

    for _ in range(population_size - 1):
        mutated = mutate(seed_mol)
        if mutated:
            population.append(Chem.MolToSmiles(mutated))
        else:
            population.append(seed_smiles)

    results = {}
    progress = []   # 🔥 ADDED (for graph)

    for gen in range(generations):
        print(f"\n=== Generation {gen} ===")

        scored = []

        for smi in population:
            score = fitness(smi, seed_smiles)
            scored.append((score, smi))

            if smi not in results or score > results[smi][0]:
                results[smi] = [score, gen]

        scored = sorted(scored, reverse=True)

        best_score, best_smi = scored[0]
        progress.append(best_score)   # 🔥 TRACK BEST SCORE

        print("Best score:", best_score)
        print("Best molecule:", best_smi)

        parents = [smi for _, smi in scored[:population_size // 2]]
        new_population = parents.copy()

        while len(new_population) < population_size * 1.5:
            p1 = Chem.MolFromSmiles(random.choice(parents))
            p2 = Chem.MolFromSmiles(random.choice(parents))

            if p1 is None or p2 is None:
                continue

            child = crossover(p1, p2)

            if child:
                child = mutate(child)

            if child is None:
                base = Chem.MolFromSmiles(random.choice(parents))
                child = mutate(base)

            if child:
                try:
                    smi = Chem.MolToSmiles(child)

                    if Chem.MolFromSmiles(smi) is None:
                        continue

                    if len(smi) < 4:
                        continue

                    new_population.append(smi)

                except:
                    continue

        population = list(set(new_population))[:population_size]

    return results, progress   # 🔥 IMPORTANT CHANGE


# ===============================
# 🔹 MAIN
# ===============================
def main():
    seed_smiles = "CCO"

    results, progress = run_ga(seed_smiles)

    unique_results = {}
    for smi, val in results.items():
        if smi not in unique_results or val[0] > unique_results[smi][0]:
            unique_results[smi] = val

    results = dict(
        sorted(unique_results.items(), key=lambda x: x[1][0], reverse=True)
    )

    results = dict(list(results.items())[:300])

    output_dir = "main/graph_ga/results"
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "results.yaml")

    with open(path, "w") as f:
        yaml.dump(results, f)

    print("\n✅ Results saved to:", path)
    print("Total UNIQUE molecules:", len(results))

    print("\nTop 5 Final:")
    for smi, val in list(results.items())[:5]:
        print(val[0], smi)


if __name__ == "__main__":
    main()