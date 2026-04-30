from __future__ import print_function

import random
from typing import List

import joblib
import numpy as np
from joblib import delayed
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol

rdBase.DisableLog('rdApp.error')

import main.graph_ga.crossover as co
import main.graph_ga.mutate as mu
from main.optimizer import BaseOptimizer

MINIMUM = 1e-10


def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int):
    population_scores = [s + MINIMUM for s in population_scores]
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]

    return np.random.choice(
        population_mol,
        p=population_probs,
        size=offspring_size,
        replace=True
    )


def reproduce(mating_pool, mutation_rate):
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)

    child = co.crossover(parent_a, parent_b)

    if child is not None:
        child = mu.mutate(child, mutation_rate)

    return child


class GB_GA_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "graph_ga"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)

        pool = joblib.Parallel(n_jobs=self.n_jobs)

        # ===============================
        # 🔥 FIX 1: FORCE INPUT DOMINANCE
        # ===============================
        if self.smi_file is not None:
            with open(self.smi_file, "r") as f:
                seed_smiles = f.read().strip()

            print("🔥 USING INPUT:", seed_smiles)

            # 🚨 STRONG FIX (IMPORTANT)
            starting_population = [seed_smiles] * config["population_size"]

        else:
            starting_population = list(
                np.random.choice(self.all_smiles, config["population_size"])
            )

        # ===============================
        # INITIAL POPULATION
        # ===============================
        population_mol = [
            Chem.MolFromSmiles(s) for s in starting_population if s
        ]

        population_mol = self.sanitize(population_mol)

        population_scores = self.oracle(
            [Chem.MolToSmiles(mol) for mol in population_mol]
        )

        patience = 0

        # ===============================
        # MAIN GA LOOP
        # ===============================
        while True:

            if len(self.oracle) > 100:
                self.sort_buffer()
                old_score = np.mean(
                    [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                )
            else:
                old_score = 0

            # ===============================
            # GENERATE OFFSPRING
            # ===============================
            mating_pool = make_mating_pool(
                population_mol,
                population_scores,
                config["population_size"]
            )

            offspring_mol = pool(
                delayed(reproduce)(
                    mating_pool,
                    config["mutation_rate"]
                )
                for _ in range(config["offspring_size"])
            )

            # ===============================
            # MERGE + CLEAN
            # ===============================
            population_mol += offspring_mol
            population_mol = self.sanitize(population_mol)

            # ===============================
            # EVALUATE
            # ===============================
            population_scores = self.oracle(
                [Chem.MolToSmiles(mol) for mol in population_mol]
            )

            # ===============================
            # SELECT BEST
            # ===============================
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(
                population_tuples,
                key=lambda x: x[0],
                reverse=True
            )[:config["population_size"]]

            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]

            # ===============================
            # EARLY STOPPING
            # ===============================
            if len(self.oracle) > 100:
                self.sort_buffer()

                new_score = np.mean(
                    [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                )

                if (new_score - old_score) < 1e-3:
                    patience += 1
                    if patience >= self.args.patience:
                        self.log_intermediate(finish=True)
                        print("✅ Converged — stopping")
                        break
                else:
                    patience = 0

            if self.finish:
                break

        # ===============================
        # 🔥 FIX 2: SAVE RESULTS (IMPORTANT)
        # ===============================
        print("💾 Saving results...")

        self.save_result(self.model_name)

        print("✅ Optimization Finished")