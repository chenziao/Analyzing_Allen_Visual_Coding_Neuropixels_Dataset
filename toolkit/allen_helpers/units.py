

GENOTYPES = {'Pvalb', 'Sst', 'Vip', 'wt'}

def get_genotype(full_genotype : str) -> str:
    """Get the abbreviated genotype: one of GENOTYPES"""
    return full_genotype.split('/', 1)[0].split('-', 1)[0]


