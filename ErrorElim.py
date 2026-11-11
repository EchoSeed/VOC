from dataclasses import dataclass, field
from typing import Optional

_UNCLASSIFIED = "<unclassified>"

@dataclass
class LearningEvent:
    iteration: int
    output: str
    is_error: bool
    error_type: Optional[str] = None
    information_value: float = field(init=False)

    def __post_init__(self):
        self.information_value = 1.0 if self.is_error else 0.1

@dataclass
class ErrorRecord:
    error_type: str
    occurrences: int = 0
    eliminated: bool = False
    elimination_iteration: Optional[int] = None
    streak: int = 0
    required_streak: int = 3
    sticky: bool = False  # sticky records never eliminate (e.g. unclassified errors)

    def register(self) -> None:
        self.occurrences += 1
        self.streak = 0
        self.eliminated = False
        self.elimination_iteration = None

    def success(self, iteration: int) -> None:
        if self.sticky or self.eliminated:
            return
        self.streak += 1
        if self.streak >= self.required_streak:
            self.eliminated = True
            self.elimination_iteration = iteration

class ErrorEliminationAgent:
    def __init__(self, performance_standard: str, required_streak: int = 3):
        self.performance_standard = performance_standard
        self.required_streak = required_streak
        self.error_registry: dict[str, ErrorRecord] = {}
        self.event_log: list[LearningEvent] = []
        self.iteration: int = 0
        self.cumulative_information: float = 0.0

    def process_event(self, output: str, is_error: bool, error_type: Optional[str] = None) -> LearningEvent:
        self.iteration += 1
        event = LearningEvent(
            iteration=self.iteration,
            output=output,
            is_error=is_error,
            error_type=error_type
        )
        self.event_log.append(event)
        self.cumulative_information += event.information_value

        if is_error:
            # Untyped errors go under a sentinel key that can't collide with a
            # real tag. That record is sticky, so it stays in active_boundaries().
            key = error_type or _UNCLASSIFIED
            record = self.error_registry.setdefault(
                key,
                ErrorRecord(
                    error_type=key,
                    required_streak=self.required_streak,
                    sticky=(key == _UNCLASSIFIED),
                ),
            )
            record.register()
        else:
            # With an error_type: targeted success test (only that boundary advances).
            # Without one: advance every boundary (sticky ones ignore it).
            if error_type:
                record = self.error_registry.get(error_type)
                if record is None:
                    raise KeyError(
                        f"targeted success for unregistered boundary {error_type!r}"
                    )
                record.success(self.iteration)
            else:
                for record in self.error_registry.values():
                    record.success(self.iteration)

        return event

    def error_rate(self) -> float:
        if not self.event_log:
            return 0.0
        errors = sum(1 for e in self.event_log if e.is_error)
        return errors / len(self.event_log)

    def active_boundaries(self) -> list[str]:
        return [k for k, v in self.error_registry.items() if not v.eliminated]
