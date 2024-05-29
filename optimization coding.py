import random
import matplotlib.pyplot as plt
import numpy as np
 
plt.show()

# A chromosome consists of the following attributes:
# Price of car
# Rate of down payment
# Cost per month
# Other Expenses
# Salary

#Example usage of the genetic algorithm
max_loan = 100000
population = 100
num_generation = 50
mutation_rate = 0.1

def genetic_algorithm(population_size):
  # Initialize the population with random chromosomes
  population = []
  for i in range(population_size):
    chromosome = [random.uniform(50000, 100000), random.uniform(0.1, 0.9),
                  random.uniform(500, 1500), random.uniform(0, 1000), random.uniform(40000, 80000)]
    population.append (chromosome)
  
# The maximum loan that can be applied for a car is represented by the variable 'x'
 
def fitness(chromosome, x):
  # Calculate the total cost of the car (including down payment, monthly costs, and other expenses)
  total_cost = chromosome[0] * chromosome[1] + chromosome[2] + chromosome[3]
 
  # Calculate the loan amount that will be needed
  loan_amount = total_cost - chromosome[0] * chromosome[1]
 
  # Check if the loan amount is greater than the maximum loan that can be applied for
  if loan_amount > x:
    # If the loan amount is greater than the maximum loan, the chromosome is not fit
    return 0
  else:
    # If the loan amount is less than or equal to the maximum loan, the chromosome is fit
    # The fitness value is calculated as the difference between the maximum loan and the loan amount
    return x - loan_amount

def select_fittest(population, fitness_values):
    # Sort the population based on their fitness values in descending order
    sorted_population = sorted(zip(population, fitness_values), key=lambda x: x[1], reverse=True)
    
    # Select the top two chromosomes with the highest fitness
    return [sorted_population[0][0], sorted_population[1][0]]

def crossover(chromosome1, chromosome2):
  # Crossover point is chosen at random
  crossover_point = random.randint (1, len(chromosome1) - 1)
 
  # The attributes from the first chromosome are copied to the offspring up to the crossover point
  offspring =chromosome1[:crossover_point]
 
  # The attributes from the second chromosome are copied to the offspring after the crossover point
  offspring.extend(chromosome2[crossover_point:])
 
  return offspring
 
def mutation(chromosome, mutation_rate):
  # The gene at the mutation rate is mutated by adding or subtracting a small random value
  chromosome [mutation_rate] += random.uniform(-0.1, 0.1)
 
  return chromosome 

def genetic_algorithm(x, population_size, num_generations):
  # Initialize the population with random chromosomes
  population = []
  for i in range(population_size):
    chromosome = [random.uniform(50000, 100000), random. uniform(0.1, 0.9),
              random.uniform(500, 1500), random.uniform(0, 1000), random.uniform(40000, 80000)]
  population.append (chromosome)
 
  #Evaluate the fitness of each chromosome in the population
  fitness_values = [fitness(chromosome, x) for chromosome in population]
 
  #Run the genetic algorithm for the specified number of generations
  for i in range(num_generations):
    #Select the fittest chromosomes for reproduction
    fittest_indexes=[i for i in range(len(fitness_values)) if fitness_values[i]==max(fitness_values)]
    fittest = [population[i] for i in fittest_indexes]
 
      #Create the offspring by performing crossover and mutation on the fittest chromosomes
    offspring = []
    for i in range(len(fittest) // 2):
      offspring.extend(crossover(fittest[i], fittest[len(fittest) - i - 1]))
 
    for i in range(len(offspring)):
      offspring[i] = mutation(offspring[i])
 
    #Evaluate the fitness of the offspring
    offspring_fitness_values = [fitness(chromosome, x) for chromosome in offspring]
 
    #Select the best individuals from the current population and the offspring to form the new population
    population = [population[i] for i in fittest_indexes]
    population.extend(offspring)
    fitness_values = [fitness_values[i] for i in fittest_indexes]
    fitness_values.extend(offspring_fitness_values)
 
    #Return the fittest chromosome in the final population
    fittest_index = fitness_values.index(max(fitness_values))
    return population[fittest_index]

#Display the output and plot the graph
best_chromosome = genetic_algorithm(max_loan, population, num_generation)
print(best_chromosome)
plt.plot(best_chromosome, label='Best Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('GA Performance over Generations')
plt.legend()
plt.show()