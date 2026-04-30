from __future__ import print_function

import os
import yaml
import random
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import tdc
from tdc.generation import MolGen


# ===============================
# 🔮 ORACLE
# ===============================
class Oracle:
    def __init__(self, args=None, mol_buffer=None):
        self.name = None
        self.evaluator = None
        self.task_label = None

        self.args = args
        self.output_dir = "main/graph_ga/results"
        os.makedirs(self.output_dir, exist_ok=True)

        self.max_oracle_calls = args.max_oracle_calls
        self.freq_log = args.freq_log

        self.mol_buffer = mol_buffer if mol_buffer is not None else {}
        self.seed_smiles = None

    def assign_evaluator(self, evaluator):
        self.evaluator = evaluator

    def sort_buffer(self):
        self.mol_buffer = dict(
            sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True)
        )

    def save_result(self, filename="results.yaml"):
        path = os.path.join(self.output_dir, filename)

        self.sort_buffer()

        with open(path, "w") as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)

        print("✅ YAML SAVED:", path)

    # ===============================
    # 🔥 SCORING FUNCTION
    # ===============================
    def score_smi(self, smi):

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 0

        smi = Chem.MolToSmiles(mol)

        base_score = float(self.evaluator(smi))

        similarity = 0
        if self.seed_smiles:
            seed_mol = Chem.MolFromSmiles(self.seed_smiles)
            if seed_mol:
                fp1 = AllChem.GetMorganFingerprintAsBitVect(seed_mol, 2)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
                similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

        # 🔥 FORCE INPUT INFLUENCE
        final_score = 0.3 * base_score + 0.7 * similarity

        self.mol_buffer[smi] = [final_score, len(self.mol_buffer)]

        return final_score

    def __call__(self, smiles_list):
        return [self.score_smi(s) for s in smiles_list]


# ===============================
# 🧠 OPTIMIZER
# ===============================
class BaseOptimizer:

    def __init__(self, args=None):
        self.args = args
        self.smi_file = args.smi_file

        self.oracle = Oracle(args=self.args)

        # Load dataset
        data = MolGen(name='ZINC')
        self.all_smiles = data.get_data()['smiles'].tolist()

        # Read input molecule
        self.seed_smiles = None
        if self.smi_file and os.path.exists(self.smi_file):
            with open(self.smi_file, "r") as f:
                self.seed_smiles = f.read().strip()

        self.oracle.seed_smiles = self.seed_smiles

    def optimize(self, oracle, config, seed=0):

        print("🔥 OPTIMIZATION STARTED")
        print("INPUT:", self.seed_smiles)

        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        self.oracle.assign_evaluator(oracle)

        # ===============================
        # 🔥 START POPULATION
        # ===============================
        if self.seed_smiles:
            population = [self.seed_smiles] * config["population_size"]
        else:
            population = list(np.random.choice(self.all_smiles, config["population_size"]))

        # ===============================
        # 🔥 EVOLUTION LOOP
        # ===============================
        for gen in range(5):
            print(f"Generation {gen}")

            scores = self.oracle(population)

            sorted_pairs = sorted(zip(scores, population), reverse=True)

            # select top
            top_k = config["population_size"] // 2
            parents = [s for _, s in sorted_pairs[:top_k]]

            new_population = []

            for smi in parents:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    new_population.append(smi)

                    try:
                        mol_copy = Chem.RWMol(mol)

                        if mol_copy.GetNumAtoms() > 5:
                            idx = random.randint(0, mol_copy.GetNumAtoms() - 1)
                            mol_copy.RemoveAtom(idx)

                        new_smi = Chem.MolToSmiles(mol_copy)
                        if new_smi:
                            new_population.append(new_smi)

                    except:
                        pass

            population = new_population[:config["population_size"]]

        # ===============================
        # 🔥 SAVE RESULTS (FINAL FIX)
        # ===============================
        self.oracle.save_result("results_generated.yaml")