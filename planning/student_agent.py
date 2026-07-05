import re
from collections import deque


class AssemblyAgent:
    """Deterministic planner for both Lab 4 domains."""

    COLORS = ("red", "blue", "orange", "yellow", "green", "purple", "white", "black")

    def solve(self, scenario_context: str, llm_engine_func=None) -> list:
        if "set of blocks" in scenario_context.lower():
            initial, goal = self._parse_blocks(scenario_context)
            return self._solve_blocks(initial, goal)

        initial, goal = self._parse_abstract(scenario_context)
        return self._solve_abstract(initial, goal)

    def _last_statement(self, scenario_context: str) -> str:
        return scenario_context.split("[STATEMENT]")[-1].split("[PLAN]")[0].lower()

    def _split_initial_goal(self, scenario_context: str) -> tuple[str, str]:
        statement = self._last_statement(scenario_context)
        initial = re.search(
            r"as initial conditions i have that, (.*?)\.\s*my goal",
            statement,
            re.S,
        )
        goal = re.search(
            r"my goal is to have that (.*?)\.\s*my plan",
            statement,
            re.S,
        )
        if not initial or not goal:
            raise ValueError("Could not parse initial state and goal")
        return initial.group(1), goal.group(1)

    def _parse_abstract(self, scenario_context: str) -> tuple[set[tuple], set[tuple]]:
        initial_text, goal_text = self._split_initial_goal(scenario_context)

        def parse(text: str) -> set[tuple]:
            facts = set()
            for x, y in re.findall(r"object ([a-z]) craves object ([a-z])", text):
                facts.add(("craves", x, y))
            for x in re.findall(r"planet object ([a-z])", text):
                facts.add(("planet", x))
            for x in re.findall(r"province object ([a-z])", text):
                facts.add(("province", x))
            for x in re.findall(r"pain object ([a-z])", text):
                facts.add(("pain", x))
            if re.search(r"\bharmony\b", text):
                facts.add(("harmony",))
            return facts

        return parse(initial_text), parse(goal_text)

    def _parse_blocks(self, scenario_context: str) -> tuple[set[tuple], set[tuple]]:
        initial_text, goal_text = self._split_initial_goal(scenario_context)

        def parse(text: str) -> set[tuple]:
            facts = set()
            for color in self.COLORS:
                if f"the {color} block is unobstructed" in text:
                    facts.add(("clear", color))
                if f"the {color} block is on the table" in text:
                    facts.add(("table", color))
            if "the hand is empty" in text:
                facts.add(("handempty",))
            for x, y in re.findall(r"the (\w+) block is on top of the (\w+) block", text):
                facts.add(("on", x, y))
            return facts

        return parse(initial_text), parse(goal_text)

    def _bfs(self, initial: set[tuple], goal: set[tuple], neighbors, max_depth: int) -> list:
        initial_state = frozenset(initial)
        if goal <= set(initial_state):
            return []

        queue = deque([(initial_state, [])])
        seen = {initial_state}
        while queue:
            state, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for action, next_state in neighbors(state):
                if next_state in seen:
                    continue
                next_path = path + [action]
                if goal <= set(next_state):
                    return next_path
                seen.add(next_state)
                queue.append((next_state, next_path))

        raise ValueError("No plan found")

    def _solve_abstract(self, initial: set[tuple], goal: set[tuple]) -> list:
        objects = sorted({item for fact in initial | goal for item in fact[1:]})

        def neighbors(state):
            state = set(state)
            for x in objects:
                if {("province", x), ("planet", x), ("harmony",)} <= state:
                    yield (
                        f"(attack {x})",
                        frozenset(
                            (state - {("province", x), ("planet", x), ("harmony",)})
                            | {("pain", x)}
                        ),
                    )

                if ("pain", x) in state:
                    yield (
                        f"(succumb {x})",
                        frozenset(
                            (state - {("pain", x)})
                            | {("province", x), ("planet", x), ("harmony",)}
                        ),
                    )

                for y in objects:
                    if x == y:
                        continue
                    if {("province", y), ("pain", x)} <= state:
                        yield (
                            f"(overcome {x} {y})",
                            frozenset(
                                (state - {("province", y), ("pain", x)})
                                | {("harmony",), ("province", x), ("craves", x, y)}
                            ),
                        )
                    if {("craves", x, y), ("province", x), ("harmony",)} <= state:
                        yield (
                            f"(feast {x} {y})",
                            frozenset(
                                (state - {("craves", x, y), ("province", x), ("harmony",)})
                                | {("pain", x), ("province", y)}
                            ),
                        )

        return self._bfs(initial, goal, neighbors, max_depth=24)

    def _solve_blocks(self, initial: set[tuple], goal: set[tuple]) -> list:
        objects = sorted({item for fact in initial | goal for item in fact[1:]})

        def neighbors(state):
            state = set(state)
            for x in objects:
                if {("clear", x), ("table", x), ("handempty",)} <= state:
                    yield (
                        f"(engage_payload {x})",
                        frozenset(
                            (state - {("clear", x), ("table", x), ("handempty",)})
                            | {("holding", x)}
                        ),
                    )

                if ("holding", x) in state:
                    yield (
                        f"(release_payload {x})",
                        frozenset(
                            (state - {("holding", x)})
                            | {("handempty",), ("table", x), ("clear", x)}
                        ),
                    )

                for y in objects:
                    if x == y:
                        continue
                    if {("clear", x), ("on", x, y), ("handempty",)} <= state:
                        yield (
                            f"(unmount_node {x} {y})",
                            frozenset(
                                (state - {("clear", x), ("on", x, y), ("handempty",)})
                                | {("holding", x), ("clear", y)}
                            ),
                        )
                    if {("holding", x), ("clear", y)} <= state:
                        yield (
                            f"(mount_node {x} {y})",
                            frozenset(
                                (state - {("holding", x), ("clear", y)})
                                | {("handempty",), ("on", x, y), ("clear", x)}
                            ),
                        )

        return self._bfs(initial, goal, neighbors, max_depth=24)
