import torch
import random


import submission2_utils as util


def genetic_algorithm(inputs, fitness_function, num_forwarded, num_generations,
                      threshold, num_df, mapping):
    generations = [get_fitnesses(inputs, fitness_function, num_df)]

    for _ in range(num_generations):
        previous = generations[-1]

        next_generation = create_generation(previous, num_forwarded)
        next_generation = \
            get_fitnesses(next_generation, fitness_function, num_df)
        pruned_next = prune_generation(
            next_generation, fitness_function, num_forwarded, num_df, mapping,
            threshold)

        generations.append(pruned_next)

    return generations


def prune_generation(generation, fitness_function, num_forwarded, num_df,
                     mapping, threshold):
    """Replace the members of the previous generation that exceeded the
    threshold with random new ones that don't.
    """
    pruned = [prev for prev in generation if prev[0] <= threshold]

    num_new = num_forwarded - len(pruned)
    if num_new == 0:
        sort_fitnesses(pruned)
        return pruned
    else:
        replacements = \
            [house for house in util.create_houses(num_new, num_df, mapping)]
        replacements = get_fitnesses(replacements, fitness_function, num_df)

        pruned.extend(replacements)
        return prune_generation(pruned, fitness_function, num_forwarded,
                                num_df, mapping, threshold)


def get_fitnesses(generation, fitness_function, num_df):
    fitnesses = []
    for house in generation:
        normalized_house = util.normalize_single(num_df, house)
        normalized_fitness = \
            fitness_function(torch.Tensor(normalized_house))[0].item()
        fitness = util.unnormalize(num_df, 'SalePrice', normalized_fitness)
        fitnesses.append((fitness, house))

    return sort_fitnesses(fitnesses)


def sort_fitnesses(fitnesses):
    fitnesses.sort(key=lambda x: x[0], reverse=True)
    return fitnesses


def create_generation(previous_generation, num_forwarded):
    sequence_length = len(previous_generation[0][1])

    next_generation = []
    # The weight is just the fitness here. A higher fitness is always going to
    # be better.
    weights = [prev[0] for prev in previous_generation]
    for _ in range(num_forwarded):
        first_parent = \
            random.choices(previous_generation, weights=weights)[0][1]
        second_parent = \
            random.choices(previous_generation, weights=weights)[0][1]

        # Need two unique parents
        while second_parent == first_parent:
            second_parent = \
                random.choices(previous_generation, weights=weights)[0][1]

        split_loc = random.randrange(sequence_length)
        child = first_parent[0:split_loc] + second_parent[split_loc:]
        next_generation.append(child)

    return next_generation