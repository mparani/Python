import random, math
from itertools import permutations
from operator import attrgetter

class Chromosome(object):
    """
    Chromosome  that encapsulates fitness and solution .
    """
    def __init__(self, genes):
        """Initialise the Chromosome."""
        self.genes = genes
        self.fit_score = 0
        self.probability_score = 0.0

    def __repr__(self):
        """
        Return in string format
        """
        return repr((self.genes, self.fit_score, self.probability_score))

    def get_fit_score(self, fit_score):
        """
        Return probability score for object - Integer
        """
        return self.fit_score

    def get_probability_score(self):
        """
        Return probability score for object - Float
        """
        return self.probability_score

    def set_fit_score(self,fit_score):
        """
        Set Fitness score for each board
        """
        self.fit_score = fit_score

    def set_probability_score(self, probability_score):
        """
        Set probability score for object - Float
        """
        self.probability_score = probability_score


class Algorithm(object):
    """
    Genetic Algorithm class to initialize the population size, assign fitness/probability score, cross-over, mutation and new generation creation
    """
    def __init__(self, nsize = 8, population_size=150, generations=1000, mutation_factor=0.2, maximise_fitness=True):
        """Instantiate the Algorithm.
        :param nsize: number of queens to GA
        :param int population_size: size of population
        :param int generations: number of generations to evolve
        :param float mutation_probability: probability of mutation operation
        """
        self.nsize = nsize
        self.population_size = population_size = population_size if (population_size and population_size <= math.factorial(nsize)) else math.factorial(nsize)
        self.initial_generation = []
        self.new_generation = []
        self.generations = generations
        self.mutation_factor = mutation_factor

    def create_initial_population(self):
        """
            Function to create initial population based on number of queens and requested size, uses permutated value to avoid single straight row/column move
        """
        seeds =  random.sample(list(permutations([x for x in range(self.nsize)], self.nsize)), self.population_size)
        population = []
        for gene in seeds:
            individual = Chromosome(gene)
            population.append(individual)
        self.initial_generation = population

    def calculate_sum_fit_score(self):
        """
        :return: sum of fitness score
        """
        return sum([x.fit_score for x in self.initial_generation])

    def mean_fit_score(self):
        """
        :return: mean of fitness score
        """
        return sum([x.fit_score for x in self.initial_generation])/len(self.initial_generation)

    def assign_fit_score(self):
        """
        Function to generate fitness score based on diagonal attack and with calculated threshold limit
        :param genoe: Single board variant based on populated values
        :return: fitness score
        """
        for individual in self.initial_generation:
            collissions = 0
            for i in range(len(individual.genes)):
                for j in range(i, len(individual.genes)):
                    if (i != j) and (abs(i - j) == abs(individual.genes[i] - individual.genes[j])):
                        collissions += 1
            individual.set_fit_score(collissions)

    def assign_probability_score(self):
        """
        Method to generate probability % for easier selection
        """
        for individual in self.initial_generation:
            individual.set_probability_score(1 - (individual.fit_score/self.calculate_sum_fit_score()))

    @staticmethod
    def crossover(parentA_genes, parentB_genes):
        """
        Crossover based on random index between concatenate based on child
        return crossed over child
        """
        crossover_index = random.randrange(1, len(parentA_genes))
        child_1a = parentA_genes[:crossover_index]
        child_1b = tuple(i for i in parentB_genes if i not in child_1a)
        childA_genes = child_1a + child_1b
        child_2a = parentB_genes[crossover_index:]
        child_2b = tuple(i for i in parentB_genes if i not in child_2a)
        childB_genes = child_2a + child_2b
        return childA_genes, childB_genes

    @staticmethod
    def mutate(childA_genes, childB_genes):
        """
        Switch the numbers based on randomly selected index for both the child
        return mutated child
        """
        mutate_idx1 = random.randrange(len(childA_genes))
        mutate_idx2 = random.randrange(len(childA_genes))
        childA_genes, childB_genes = list(childA_genes), list(childB_genes)
        childA_genes[mutate_idx1], childA_genes[mutate_idx2] = childA_genes[mutate_idx2], childA_genes[mutate_idx1]
        childB_genes[mutate_idx1], childB_genes[mutate_idx2] = childB_genes[mutate_idx2], childB_genes[mutate_idx1]
        return tuple(childA_genes), tuple(childB_genes)

    def rank_initial_population(self):
        """
        Sort based on probability score - and paired by most fitted
        """
        self.initial_generation.sort(key=attrgetter('probability_score'), reverse=True)

    def create_new_generation(self):
        """
        Create new generation based on paired-fitted selection process
        Execute Crossover based on parent genes
        Based on mutation probability factor, apply mutation
        Assign the newly generated as initial population
        """
        x = 0
        new_generation = []
        while x < (math.floor(len(self.initial_generation))/2):
            parentA_genes, parentB_genes = self.initial_generation[x].genes, self.initial_generation[x + 1].genes
            childA_genes, childB_genes = self.crossover(parentA_genes, parentB_genes)
            if random.random() < self.mutation_factor:
                childA_genes, childB_genes = self.mutate(childA_genes, childB_genes)
            new_generation.append(childA_genes)
            new_generation.append(childB_genes)
            x += 1
        self.initial_generation = [Chromosome(t) for t in new_generation]

    def generate_initial_population_and_score(self):
        """
           Function to create new population based on number of queens and requested size, uses permutated value to avoid single straight row/column move
        """
        self.create_initial_population()
        self.assign_fit_score()
        self.assign_probability_score()
        self.rank_initial_population()

    def generate_new_population_and_score(self):
        """
           Function to create new population based on initial or previous ranked population, enable cross over , mutate and assign to initial population
        """
        self.create_new_generation()
        self.assign_fit_score()
        self.assign_probability_score()
        self.rank_initial_population()


if __name__ == "__main__":
     num_queens = 8
     population_size = 150
     mutation_factor = 0.5
     threshold_limit = 1000
     qconfig = (7, 1, 4, 2, 0, 6, 3, 5)
     LOOP = 1
     MATCH_FOUND = False
     ga = Algorithm(num_queens,population_size, threshold_limit, mutation_factor)
     ga.generate_initial_population_and_score()
     while LOOP <= threshold_limit:
        if not MATCH_FOUND:
            ga.generate_new_population_and_score()
            mean_fit_score = ga.mean_fit_score()
            print("Generation : {},Genes : {}, Average Population Fitness Score : {} ".format(LOOP, qconfig, mean_fit_score))
            for chromosome in ga.initial_generation:
                if chromosome.genes == qconfig:
                    print("### QCONFIG MATCHED AT GENERATION : {} WITH MEAN FIT SCORE : {} ###".format(LOOP,mean_fit_score ))
                    print("CHROMOSOME - GENES : {}".format(chromosome.genes))
                    print("CHROMOSOME - FITNESS SCORE : {}".format(chromosome.fit_score))
                    print("CHROMOSOME - PROBABILITY SCORE : {}".format(chromosome.probability_score))
                    print("### QCONFIG MATCHED ###")
                    MATCH_FOUND = True
                    break
        LOOP += 1

