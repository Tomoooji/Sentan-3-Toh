import numpy as np
import matplotlib.pyplot as plt

from execute_GA import GeneticAlgorithmAligner as GA1
from execute_GAv2 import GeneticAlgorithmAligner as GA2

param_table = [
    {"pdb1":"1jl9.pdb", "pdb2":"1jl9m.pdb", "aln":"1jl9.aln.fasta", "pop":100, "gen":100, "rec":0.7, "mut":0.7},
    {"pdb1":"1jl9.pdb", "pdb2":"1jl9m.pdb", "aln":"1jl9.aln.fasta", "pop":3,   "gen":100, "rec":0.7, "mut":0.7},
    {"pdb1":"1jl9.pdb", "pdb2":"1jl9m.pdb", "aln":"1jl9.aln.fasta", "pop":100, "gen":5,   "rec":0.7, "mut":0.7},
    {"pdb1":"1jl9.pdb", "pdb2":"1jl9m.pdb", "aln":"1jl9.aln.fasta", "pop":100, "gen":100, "rec":0,   "mut":0.7},
    {"pdb1":"1jl9.pdb", "pdb2":"1jl9m.pdb", "aln":"1jl9.aln.fasta", "pop":100, "gen":100, "rec":0.7, "mut":0  },
    {"pdb1":"3c9a.pbd", "pdb2":"1jl9.pdb",  "aln":"seq.aln.fasta", "pop":100, "gen":100, "rec":0.7, "mut":0.7},
    {"pdb1":"3c9a.pbd", "pdb2":"1jl9.pdb",  "aln":"seq.aln.fasta", "pop":3,   "gen":100, "rec":0.7, "mut":0.7},
    {"pdb1":"3c9a.pbd", "pdb2":"1jl9.pdb",  "aln":"seq.aln.fasta", "pop":100, "gen":5,   "rec":0.7, "mut":0.7},
    {"pdb1":"3c9a.pbd", "pdb2":"1jl9.pdb",  "aln":"seq.aln.fasta", "pop":100, "gen":100, "rec":0,   "mut":0.7},
    {"pdb1":"3c9a.pbd", "pdb2":"1jl9.pdb",  "aln":"seq.aln.fasta", "pop":100, "gen":100, "rec":0.7, "mut":0  },
    {"pdb1":"3c9a.pbd", "pdb2":"1jl9.pdb",  "aln":"seq.aln.fasta", "pop":500, "gen":100, "rec":0.7, "mut":0.7}
]

for param in param_table:
    for i in range(5):
        ga1=GA1(
            param["pdb1"], param["pdb2"], param["aln"], param["pop"], param["gen"], param["mut"], param["rec"], i, False
        )
        population = ga1.prepare_ga()
        for g in range(ga1.gen_num):
            mutated = ga1.mutation(population)
            recombinated = ga1.recombination(mutated)
            fitness_vals = np.array([ga1.calc_fitness(ind) for ind in recombinated])
            population = ga1.selection(recombinated, fitness_vals, g)
        ga1.output_results(population)
        print(ga1.recd[-1])

for i in range(5):
    ga2=GA2(
        param_table[-1]["pdb1"], param_table[-1]["pdb2"], param_table[-1]["aln"], param_table[-1]["pop"], param_table[-1]["gen"], param_table[-1]["mut"], param_table[-1]["rec"], i, False
    )
    population = ga2.prepare_ga()
    for g in range(ga2.gen_num):
        mutated = ga2.mutation(population)
        recombinated = ga2.recombination(mutated)
        fitness_vals = np.array([ga2.calc_fitness(ind) for ind in recombinated])
        population = ga2.selection(recombinated, fitness_vals, g)
    ga2.output_results(population)
    print(ga2.recd[-1])
    
