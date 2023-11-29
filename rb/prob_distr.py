class ProbDistr(dict[str, float]):
    def __init__(self, data: dict[str, float]) -> None:
        super().__init__(data)

    @staticmethod
    def from_counts(counts: dict[str, int]) -> "ProbDistr":
        shots = sum(counts.values())
        return ProbDistr({k: v / shots for k, v in counts.items()})

    def to_counts(self, shots: int) -> dict[str, int]:
        return {k: round(v * shots) for k, v in self.items()}
