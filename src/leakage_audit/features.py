"""The feature specification, and the pipeline graph built from it.

Every field here exists because it is something a feature definition knows and a scored
model does not. A trained model sees a column of numbers; it has no idea that the column
was built from a window straddling the prediction moment, or that the table it came from
does not land until two days later. Those facts live in the definition, which is why the
audit reads definitions rather than scores.

Offsets are integer periods relative to the prediction moment, and negative means the past.
A window of (-7, -1) is the week before the prediction. A window of (-3, 3) is centred on
it, and is the shape of mistake this repository exists to find.
"""

from __future__ import annotations

from dataclasses import dataclass, field

EVENTS = "events"
TARGET = "target"
SOURCES = (EVENTS, TARGET)


class SpecError(ValueError):
    """Raised when a feature or pipeline is malformed, as opposed to merely leaking."""


@dataclass(frozen=True)
class FeatureSpec:
    """One feature, described the way a feature store describes it.

    Attributes:
        name: unique identifier within the pipeline.
        source: which table the raw values come from. A feature reading from the target
            table is not automatically a leak, because a lagged target is a legitimate and
            common feature, but it is where leaks concentrate.
        window_start: earliest offset the aggregation reads, relative to prediction time.
        window_end: latest offset it reads. This is the field that decides everything.
        ingestion_lag: periods between an event happening and its row being queryable.
            A window ending yesterday is useless if yesterday's rows arrive tomorrow.
        depends_on: other features this one is computed from. A feature inherits the
            availability limits of everything upstream of it, which is the part people
            miss when a pipeline is three joins deep.
        description: free text, carried into the report so a violation names something a
            human recognises.
    """

    name: str
    source: str = EVENTS
    window_start: int = -7
    window_end: int = -1
    ingestion_lag: int = 0
    depends_on: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise SpecError("a feature needs a name")
        if self.source not in SOURCES:
            raise SpecError(f"unknown source {self.source!r}, expected one of {SOURCES}")
        if self.window_start > self.window_end:
            raise SpecError(
                f"{self.name}: window_start {self.window_start} is after window_end "
                f"{self.window_end}, which describes no window at all"
            )
        if self.ingestion_lag < 0:
            raise SpecError(f"{self.name}: ingestion_lag cannot be negative")
        if self.name in self.depends_on:
            raise SpecError(f"{self.name}: a feature cannot depend on itself")

    @property
    def latest_offset_read(self) -> int:
        """The newest moment this feature's own window touches."""
        return self.window_end

    @property
    def available_from(self) -> int:
        """The earliest offset at which this feature's own window is fully queryable.

        A window ending at offset e, over a source lagging by l, is not computable until
        offset e + l. Comparing that against zero is the whole availability test.
        """
        return self.window_end + self.ingestion_lag

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "ingestion_lag": self.ingestion_lag,
            "depends_on": list(self.depends_on),
            "description": self.description,
        }


@dataclass(frozen=True)
class Pipeline:
    """A set of feature specs, plus the graph they form."""

    features: tuple[FeatureSpec, ...]
    horizon: int = 1
    _by_name: dict[str, FeatureSpec] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        names = [f.name for f in self.features]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise SpecError(f"duplicate feature names: {sorted(duplicates)}")
        if self.horizon < 1:
            raise SpecError("horizon must be at least one period ahead")
        object.__setattr__(self, "_by_name", {f.name: f for f in self.features})

    def __len__(self) -> int:
        return len(self.features)

    def get(self, name: str) -> FeatureSpec:
        if name not in self._by_name:
            raise SpecError(f"no feature named {name!r} in this pipeline")
        return self._by_name[name]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features)

    def missing_dependencies(self) -> tuple[tuple[str, str], ...]:
        """Edges pointing at features the pipeline does not contain.

        Reported rather than raised. A pipeline assembled from several repositories often
        has dangling edges, and refusing to audit it at all would mean the tool is unusable
        exactly where it is most needed.
        """
        missing = []
        for feature in self.features:
            for dependency in feature.depends_on:
                if dependency not in self._by_name:
                    missing.append((feature.name, dependency))
        return tuple(missing)

    def find_cycle(self) -> tuple[str, ...]:
        """Return one cycle in the dependency graph, or an empty tuple.

        Depth-first with an explicit stack rather than recursion, because a real feature
        graph can be deeper than the interpreter's recursion limit and crashing on a large
        pipeline would be a poor way to report a cycle.
        """
        WHITE, GREY, BLACK = 0, 1, 2
        colour = dict.fromkeys(self._by_name, WHITE)

        for start in self._by_name:
            if colour[start] != WHITE:
                continue
            stack: list[tuple[str, list[str]]] = [(start, [start])]
            while stack:
                node, path = stack.pop()
                if colour[node] == GREY:
                    colour[node] = BLACK
                    continue
                colour[node] = GREY
                stack.append((node, path))
                for dependency in self._by_name[node].depends_on:
                    if dependency not in self._by_name:
                        continue
                    if colour[dependency] == GREY:
                        cut = path.index(dependency) if dependency in path else 0
                        return tuple([*path[cut:], dependency])
                    if colour[dependency] == WHITE:
                        stack.append((dependency, [*path, dependency]))
        return ()

    def ancestors(self, name: str) -> tuple[str, ...]:
        """Every feature upstream of this one, transitively.

        Used by the audit because availability is inherited: a feature computable on paper
        is not computable in practice if anything it is built from is still in the future.
        """
        seen: set[str] = set()
        stack = list(self.get(name).depends_on)
        while stack:
            current = stack.pop()
            if current in seen or current not in self._by_name:
                continue
            seen.add(current)
            stack.extend(self._by_name[current].depends_on)
        return tuple(sorted(seen))

    def as_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "features": [f.as_dict() for f in self.features],
        }
