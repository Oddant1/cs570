import torch
import random

import submission2_utils as util

# The percentage chance that we will mutate a given column
MUTATION_CHANCE = 1
# The maximum amount we will mutate a continuous column by (multiply this by
# value then add or subtract that amounts)
MAX_MUTATION_CHANGE = .05


def genetic_algorithm(inputs, fitness_function, num_forwarded, num_generations,
                      threshold, num_df, mapping, reverse_mapping):
    """Takes an initial generation then uses it as a basis to generate future
    generations.
    """
    generations = [get_fitnesses(inputs, fitness_function, num_df)]
    pruned_generations = []

    for _ in range(num_generations):
        previous = generations[-1]
        pruned_previous, pruned_out = prune_generation(
            previous, fitness_function, num_df, mapping, threshold)
        pruned_generations.append(pruned_out)

        next_generation = create_generation(pruned_previous, num_forwarded)
        mutated = mutate(next_generation, reverse_mapping)
        fitness_ranked = \
            get_fitnesses(mutated, fitness_function, num_df)
        generations.append(fitness_ranked)

    return generations, pruned_generations


def get_fitnesses(generation, fitness_function, num_df):
    """Get all fitnesses of the current generation.
    """
    fitnesses = []
    for house in generation:
        normalized_house = util.normalize_single(num_df, house.fields)
        normalized_fitness = \
            fitness_function(torch.Tensor(normalized_house))[0].item()
        fitness = util.unnormalize(num_df, 'SalePrice', normalized_fitness)
        house.fitness = fitness
        fitnesses.append(house)

    return sort_fitnesses(fitnesses)


def prune_generation(generation, fitness_function, num_df, mapping, threshold):
    """Keep only the members of the current generation that do not exceed the
    cost threshold. If we end up with fewer than two valid parents we generate
    new houses that we ensure are under the threshold until we have two valid
    parents.
    """
    remain, pruned = _prune_helper(generation, threshold)

    # We need at least two valid parents. We are in pretty bad shape if we
    # have one and even worse if we have 0, so we need to create new ones.
    while (num_new := 2 - len(remain)) > 0:
        replacements = [house for house in util.create_houses(num_new, num_df)]
        replacements = get_fitnesses(replacements, fitness_function, num_df)
        remain.extend(replacements)
        remain, new_pruned = _prune_helper(remain, threshold)
        pruned.extend(new_pruned)

    sort_fitnesses(remain)
    sort_fitnesses(pruned)
    return remain, pruned


def _prune_helper(to_prune, threshold):
    remain = []
    pruned = []

    for house in to_prune:
        if house.fitness <= threshold:
            remain.append(house)
        else:
            pruned.append(house)

    return remain, pruned


def create_generation(pruned_previous, num_forwarded):
    """Take the previous generation and create the next generation.
    """
    sequence_length = len(pruned_previous[0].fields)

    next_generation = []
    # Once we get here, we will have pruned out unfit members of the previous
    # generation. We are guaranteed to have at least two valid parents here.
    weights = \
        [prev.fitness * (1 / (idx + 1)) for idx, prev in \
         enumerate(pruned_previous)]
    for _ in range(num_forwarded):
        first_parent = \
            random.choices(pruned_previous, weights=weights)[0].fields
        second_parent = \
            random.choices(pruned_previous, weights=weights)[0].fields

        # Need two unique parents.
        while second_parent == first_parent:
            second_parent = \
                random.choices(pruned_previous, weights=weights)[0].fields

        split_loc = random.randrange(sequence_length)
        child_fields = first_parent[0:split_loc] + second_parent[split_loc:]
        child = util.House(0, 0, child_fields)
        next_generation.append(child)

    return next_generation


def mutate(generation, reverse_mapping):
    """Mutate members of the current generation.
    """
    mutated = generation.copy()

    for house in mutated:
        for idx, value in enumerate(house.fields):
            if random.uniform(0, 100) < MUTATION_CHANCE:
                if idx in reverse_mapping.keys():
                    # Randomly choose a new category.
                    new = random.randrange(reverse_mapping[idx]['MAX_VAL'])
                else:
                    change = value * random.uniform(
                        -MAX_MUTATION_CHANGE, MAX_MUTATION_CHANGE)
                    new = value + change
                    # All the data is in integers, so we round
                    new = round(new)
                house.fields[idx] = new

    return mutated


def sort_fitnesses(generation):
    """Sort the given generation by fitness high to low.
    """
    generation.sort(key=lambda x: x.fitness, reverse=True)
    return generation
