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

for pram in param_table:
    ...