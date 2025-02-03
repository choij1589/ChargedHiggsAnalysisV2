# In this script, the codes for hyperparameter optimization using genetic algorithm are provided.
# The method for selection, crossover, and mutation is from
# https://link.springer.com/content/pdf/10.1007/s11042-020-10139-6.pdf

import os
import logging
import random
import numpy as np
import pandas as pd
from itertools import product

class GeneticModule:
    def __init__(self):
        self.config_space = []  # hyperparameter space
        self.pool = []          # all possible configurations
        self.population = []    # current population

    def setConfigSpace(self, genes):
        """Appends a new set of gene values to the configuration space and updates the pool of all possible chromosomes accordingly. This method expands the search space with additional configuration possibilities."""
        self.config_space.append(genes)
        self.pool = list(product(*self.config_space))

    def generatePopulation(self, criteria=None):
        """Generates a new population from the pool. It can optionally filter the pool using a provided criteria function to include only those chromosomes that meet certain conditions."""
        if criteria:
            self.pool = list(filter(criteria, self.pool))
        self.population = [{'chromosome': ch, 'fitness': None} for ch in random.sample(self.pool, len(self.pool))]

    def randomGeneration(self, n_population):
        """Samples a specified number (n_population) of individuals randomly from the current population to create a new generation. This method is typically used to initialize the population at the start of the algorithm."""
        self.population = random.choices(self.population, k=n_population)

    def updatePopulation(self, signal, background, channel, metric="loss/valid", read_from=None):
        """Updates the fitness values of each individual in the population based on their performance metrics, which are read from CSV files generated from simulations or experiments. The paths to these files are constructed dynamically using individual chromosome data."""
        for idx, individual in enumerate(self.population):
            csv_path = f"{read_from}/ParticleNet-model{idx}.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                individual['model'] = f"ParticleNet-model{idx}"
                individual['fitness'] = float(df.sort_values(metric).iloc[0][metric])
            else:
                raise FileNotFoundError(f"CSV file not found for model: {csv_path}")

    def rankSelection(self):
        """Selects two unique parents for crossover using rank-based roulette wheel selection. This method involves ranking individuals by fitness and then randomly selecting them with probabilities proportional to their ranks."""
        sorted_pop = sorted(self.population, key=lambda x: x['fitness'])
        ranks = list(range(1, len(sorted_pop)+1))
        total_rank = sum(ranks)
        selection_probs = [rank / total_rank for rank in reversed(ranks)]

        # Select first parent
        p1 = random.choices(sorted_pop, weights=selection_probs)[0]
        # Select second parent
        p2 = p1
        while p2 == p1:
            p2 = random.choices(sorted_pop, weights=selection_probs)[0]
        
        return [p1, p2]

    def uniformCrossover(self, parent1, parent2):
        """Performs a uniform crossover between two parent chromosomes to produce a child chromosome. Each gene in the child’s chromosome is independently chosen from one of the corresponding genes of the parents with equal probability."""
        child_chromosome = tuple(random.choice(pair) for pair in zip(parent1['chromosome'], parent2['chromosome']))
        if child_chromosome in self.pool:
            return {'chromosome': child_chromosome, 'fitness': None}
        return self.uniformCrossover(*self.rankSelection())  # retry with new parents

    def displacementMutation(self, child, thresholds):
        """Mutates a child’s chromosome by randomly altering some of its genes based on specified mutation thresholds. This introduces genetic diversity and helps explore new areas of the solution space."""
        mutated = tuple(random.choice(self.config_space[i]) if random.random() > thresh else gene
                        for i, (gene, thresh) in enumerate(zip(child['chromosome'], thresholds)))
        if mutated in self.pool:
            return {'chromosome': mutated, 'fitness': None}
        return self.displacementMutation(child, thresholds)  # retry mutation

    def evolution(self, thresholds, ratio):
        """Drives the evolution process by using a combination of selection, crossover, and mutation to produce a new generation. This function manages the balance of keeping strong solutions (parents) and introducing new solutions (children)."""
        parents = sorted(self.population, key=lambda x: x['fitness'])
        n_birth = int(len(parents) * ratio)
        children = []

        while len(children) < n_birth:
            p1, p2 = self.rankSelection()
            child = self.uniformCrossover(p1, p2)
            mutation = self.displacementMutation(child, thresholds)
            if mutation['chromosome'] not in [c['chromosome'] for c in children]:
                children.append(mutation)

        self.population = parents[:len(self.population)-n_birth] + children

    def bestChromosome(self):
        return min(self.population, key=lambda x: x['fitness'])

    def meanFitness(self):
        return np.mean([x['fitness'] for x in self.population if x['fitness'] is not None])

    def savePopulation(self, path):
        #os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame([{ 'model': p['model'], 'chromosome': p['chromosome'], 'fitness': p['fitness']} for p in self.population])
        df.to_csv(path, index=False)
