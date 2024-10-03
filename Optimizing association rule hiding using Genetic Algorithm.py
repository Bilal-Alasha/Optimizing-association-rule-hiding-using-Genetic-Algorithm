import os
import random
import math
import itertools
from functools import reduce

# Simple DB
transactions = [
    {'Bread', 'Butter'},
    {'Bread', 'Milk'},
    {'Milk', 'Butter'},
    {'Bread', 'Butter', 'Milk'},
    {'Bread'},
    {'Butter', 'Milk'},
]

# Support function (Alpha)
def support(itemset, transactions):
    return sum(1 for t in transactions if itemset.issubset(t)) / len(transactions)

# Confidence function (Beta)
def confidence(lhs, rhs, transactions):
    return support(lhs.union(rhs), transactions) / support(lhs, transactions)

# Apriori algorithm to find frequent itemsets 
def apriori(transactions, min_support):
    items = sorted(reduce(set.union, transactions))
    candidate_itemsets = [frozenset([item]) for item in items]
    frequent_itemsets = []

    while candidate_itemsets:
        valid_itemsets = [itemset for itemset in candidate_itemsets if support(itemset, transactions) >= min_support]
        frequent_itemsets.extend(valid_itemsets)
        candidate_itemsets = [itemset1.union(itemset2) for itemset1, itemset2 in itertools.combinations(valid_itemsets, 2)
                              if len(itemset1.union(itemset2)) == len(itemset1) + 1]
        candidate_itemsets = list(set(candidate_itemsets))  # Remove duplicates
    return frequent_itemsets

# checks the confidence of the all the frequent_itemsets lhs => rhs and adds it to the list of rules if it meets the min_confidence threshold
def generate_rules(frequent_itemsets, transactions, min_confidence):
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) > 1:
            subsets = list(itertools.chain.from_iterable(itertools.combinations(itemset, r) for r in range(1, len(itemset))))
            for lhs_tuple in subsets:
                lhs = frozenset(lhs_tuple)
                rhs = itemset - lhs
                if rhs and confidence(lhs, rhs, transactions) >= min_confidence:
                    rules.append((lhs, rhs, confidence(lhs, rhs, transactions)))
    return rules

# Find sensitive rule with the highest confidence
# (we can run the code again or stop this part if we want all or more than one rule to be effected)
def find_sensitive_rule(rules):
    sorted_rules = sorted(rules, key=lambda r: r[2], reverse=True)
    if sorted_rules:
        return sorted_rules[0]  # Highest confidence rule
    return None

# Number of transactions to modify (Delta)
def compute_delta(lhs, rhs, transactions, MCT):
    support_lhs_rhs = support(lhs.union(rhs), transactions)
    support_rhs = support(rhs, transactions)
    delta = support_lhs_rhs - math.floor(MCT * support_rhs)
    return max(0, delta * len(transactions)) #return the number of transactions as a fixed number 

# Updated fitness function (thanks gpt)
def fitness(selected_transactions, DB_prime, non_sensitive_rules, sensitive_rule, delta):
    modified_db = [t for t in DB_prime if t not in selected_transactions]
    
    lhs, rhs, _ = sensitive_rule
    sensitive_support = support(lhs.union(rhs), modified_db)

    if sensitive_support >= delta:
        return float('-inf')  # Do not take this as a solutions
    
    valid_rules = 0
    for lhs, rhs, _ in non_sensitive_rules:
        if support(lhs.union(rhs), modified_db) > 0:
            valid_rules += 1

    return valid_rules  # Higher is better lol , represents fewer lose of non-sensitive rules

# Genetic algorithm implementation (thanks gpt)
def genetic_algorithm(DB_prime, candidate_transactions, delta, non_sensitive_rules, sensitive_rule, pop_size=20, generations=100, mutation_rate=0.1):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'generation_progress.txt')
    
    population = [random.sample(candidate_transactions, min(len(candidate_transactions), int(delta))) for _ in range(pop_size)]
    
    # Open the file to log the progress
    with open(file_path, 'w') as file:
        for generation in range(generations):
            population_fitness = [(individual, fitness(individual, DB_prime, non_sensitive_rules, sensitive_rule, delta)) for individual in population]

            population_fitness = [ind_fit for ind_fit in population_fitness if ind_fit[1] != float('-inf')]

            if not population_fitness:
                break  # All solutions are invalid, stop the algorithm

            population_fitness.sort(key=lambda x: x[1], reverse=True)

            # Log current progress to the file
            best_individual, best_fitness = population_fitness[0]
            file.write(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Transactions to modify: {best_individual}\n")

            # Select parents and perform crossover and mutation (thanks gpt)
            parents = [individual for individual, _ in population_fitness[:pop_size // 2]]
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = crossover(parents[i], parents[i+1])
                    offspring.extend([child1, child2])

            offspring = [mutate(child, candidate_transactions, mutation_rate) for child in offspring]

            population = parents + offspring

        best_individual = max(population, key=lambda individual: fitness(individual, DB_prime, non_sensitive_rules, sensitive_rule, delta))
        return best_individual


# Crossover function for GA (thanks gpt)
def crossover(parent1, parent2):
    crossover_point = len(parent1) // 2
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation function for GA (thanks gpt)
def mutate(individual, candidate_transactions, mutation_rate):
    if random.random() < mutation_rate:
        index = random.randint(0, len(individual) - 1)
        new_transaction = random.choice(candidate_transactions)
        individual[index] = new_transaction
    return individual

# Main algorithm: Modify the database to hide sensitive rules
def modify_database(transactions, rules, sensitive_rules, MST, MCT):
    DB_prime = transactions.copy()
    SARcounter = 0
    non_sensitive_rules = [rule for rule in rules if rule not in sensitive_rules]

    while SARcounter < len(sensitive_rules):
        SARcounter += 1
        SAR = sensitive_rules[SARcounter - 1]

        # Print the identified sensitive rule
        print(f"Sensitive Rule: {SAR[0]} => {SAR[1]}, Confidence = {SAR[2]}")

        rhs_items = list(SAR[1])
        item_supports = {item: support({item}, DB_prime) for item in rhs_items}

        victim_item = min(item_supports, key=item_supports.get)

        delta = compute_delta(SAR[0], SAR[1], DB_prime, MCT)

        candidate_transactions = [t for t in DB_prime if victim_item in t]

        # Run the genetic algorithm to select transactions to modify
        selected_transactions = genetic_algorithm(DB_prime, candidate_transactions, delta, non_sensitive_rules, SAR)

        # Print selected transactions to modify
        print(f"Selected Transactions to Modify: {selected_transactions}")

        DB_prime = [t for t in DB_prime if t not in selected_transactions]

    return DB_prime

# Example usage
min_support = 0.5
min_confidence = 0.7
MST = 0.4  # Minimum Support Threshold
MCT = 0.6  # Minimum Confidence Threshold

frequent_itemsets = apriori(transactions, min_support)
rules = generate_rules(frequent_itemsets, transactions, min_confidence)

sensitive_rules = [find_sensitive_rule(rules)]

modified_transactions = modify_database(transactions, rules, sensitive_rules, MST, MCT)

# Print final modified database
print("\nOriginal Transactions:", transactions)
print("Modified Transactions:", modified_transactions)
