import csv
import time
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, Set


def load_function_counts(file_path: str) -> Dict[str, Dict[str, int]]:
    """Load function counts from CSV file."""
    function_counts = defaultdict(lambda: defaultdict(int))
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for function, conversation, count in reader:
            function_counts[conversation][function] = int(count)
    return function_counts


def find_common_groupings(function_counts: Dict[str, Dict[str, int]], min_support: int = 2, max_group_size: int = 5) -> \
        List[Tuple[Set[str], int]]:
    """Find common function groupings across conversations, with optimizations."""
    groupings = defaultdict(int)
    total_conversations = len(function_counts)

    for idx, (conversation, functions) in enumerate(function_counts.items(), 1):
        if idx % 10 == 0:  # Progress update every 10 conversations
            print(f"Processing conversation {idx}/{total_conversations}")

        present_functions = list(functions.keys())
        for r in range(2, min(len(present_functions) + 1, max_group_size + 1)):
            for group in combinations(present_functions, r):
                groupings[frozenset(group)] += 1

    common_groupings = [(set(group), count) for group, count in groupings.items() if count >= min_support]
    return sorted(common_groupings, key=lambda x: (-x[1], tuple(sorted(x[0]))))


def merge_overlapping_groups(groupings: List[Tuple[Set[str], int]]) -> List[Tuple[Set[str], int]]:
    """Merge overlapping groups and combine their counts."""
    merged = []
    for group, count in groupings:
        merged_group = group
        merged_count = count
        for existing_group, existing_count in merged:
            if group & existing_group:  # If there's an overlap
                merged_group |= existing_group  # Merge the groups
                merged_count = min(count, existing_count)  # Take the minimum count
                merged.remove((existing_group, existing_count))
        merged.append((merged_group, merged_count))
    return sorted(merged, key=lambda x: (-x[1], tuple(sorted(x[0]))))


def print_and_save_groupings(groupings: List[Tuple[Set[str], int]], output_file: str):
    """Print common function groupings and save to file."""
    print("\nCommon Function Groupings:")
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Functions', 'Occurrence Count'])

        for group, count in groupings:
            functions_str = ', '.join(sorted(group))
            print(f"Functions: {functions_str}")
            print(f"Appeared together in {count} conversations")
            print()

            writer.writerow([functions_str, count])

    print(f"Results saved to {output_file}")


def main():
    input_file = 'function_counts.csv'
    output_file = 'common_function_groupings_compressed.csv'
    min_support = 2
    max_group_size = 10  # Increased to allow for larger merged groups

    start_time = time.time()
    function_counts = load_function_counts(input_file)
    print(f"Loaded data from {len(function_counts)} conversations")

    common_groupings = find_common_groupings(function_counts, min_support, max_group_size)
    merged_groupings = merge_overlapping_groups(common_groupings)

    if merged_groupings:
        print_and_save_groupings(merged_groupings, output_file)
    else:
        print(f"No common groupings found with minimum support of {min_support}")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
