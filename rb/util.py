def counts_to_probs(
    all_counts: dict[str, int] | list[dict[str, int]]
) -> list[dict[str, float]]:
    all_counts = all_counts if isinstance(all_counts, list) else [all_counts]
    return [
        {state: count / sum(counts.values()) for state, count in counts.items()}
        for counts in all_counts
    ]
