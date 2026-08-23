"""Knowledge organization: concept graphs, notebooks and retrieval."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Concept:
    """A named node in the knowledge graph with optional summary text."""

    name: str
    summary: str = ""


@dataclass(frozen=True)
class Relation:
    """A typed directed edge between two concepts."""

    source: str
    target: str
    label: str = "relates-to"


@dataclass
class Note:
    """A free-form research note tagged with concepts."""

    title: str
    body: str
    tags: frozenset[str] = field(default_factory=frozenset)
    linked_concepts: tuple[str, ...] = ()


class KnowledgeGraph:
    """Typed directed concept graph with traversal utilities."""

    def __init__(self) -> None:
        self.concepts: dict[str, Concept] = {}
        self.relations: list[Relation] = []
        self._adjacency: dict[str, set[str]] = {}

    def add_concept(self, concept: Concept) -> Concept:
        """Register a concept; re-adding replaces the previous entry."""
        self.concepts[concept.name] = concept
        self._adjacency.setdefault(concept.name, set())
        return concept

    def add_relation(self, relation: Relation) -> Relation:
        """Add a directed labelled edge; both endpoints must exist."""
        for endpoint in (relation.source, relation.target):
            if endpoint not in self.concepts:
                msg = f"unknown concept: {endpoint}"
                raise ValueError(msg)
        self.relations.append(relation)
        self._adjacency[relation.source].add(relation.target)
        return relation

    def neighbors(self, name: str, direction: str = "outgoing") -> list[str]:
        """Direct neighbours of a concept ('outgoing', 'incoming' or 'both')."""
        if name not in self.concepts:
            msg = f"unknown concept: {name}"
            raise ValueError(msg)
        outgoing = sorted(self._adjacency.get(name, set()))
        if direction == "outgoing":
            return outgoing
        incoming = sorted({r.source for r in self.relations if r.target == name})
        if direction == "incoming":
            return incoming
        if direction == "both":
            return sorted(set(outgoing) | set(incoming))
        msg = f"direction must be outgoing/incoming/both: {direction!r}"
        raise ValueError(msg)

    def shortest_path(self, start: str, goal: str) -> list[str]:
        """Fewest-hops path via BFS; empty list when unreachable."""
        if start not in self.concepts or goal not in self.concepts:
            msg = "start and goal must be known concepts"
            raise ValueError(msg)
        if start == goal:
            return [start]
        visited = {start}
        parents: dict[str, str] = {}
        queue: deque[str] = deque([start])
        while queue:
            current_node = queue.popleft()
            for neighbor in sorted(self._adjacency.get(current_node, ())):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                parents[neighbor] = current_node
                if neighbor == goal:
                    path = [goal]
                    while path[-1] != start:
                        path.append(parents[path[-1]])
                    return list(reversed(path))
                queue.append(neighbor)
        return []

    def transitive_closure(self, start: str) -> set[str]:
        """Every node reachable from ``start`` by any number of hops."""
        if start not in self.concepts:
            msg = f"unknown concept: {start}"
            raise ValueError(msg)
        seen: set[str] = set()
        queue: deque[str] = deque([start])
        while queue:
            current_node = queue.popleft()
            for neighbor in self._adjacency.get(current_node, ()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        seen.discard(start)
        return seen

    def has_cycle(self) -> bool:
        """Detect whether the directed graph contains a cycle."""
        indegree = {name: 0 for name in self.concepts}
        for relation in self.relations:
            indegree[relation.target] += 1
        ready = deque(name for name, degree in indegree.items() if degree == 0)
        processed = 0
        while ready:
            current_node = ready.popleft()
            processed += 1
            for neighbor in self._adjacency.get(current_node, ()):
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    ready.append(neighbor)
        return processed != len(self.concepts)


class Notebook:
    """Tagged research notes that can reference graph concepts."""

    def __init__(self) -> None:
        self.notes: list[Note] = []

    def add(
        self,
        title: str,
        body: str,
        tags: list[str] | None = None,
        linked_concepts: list[str] | None = None,
    ) -> Note:
        note = Note(
            title=title,
            body=body,
            tags=frozenset(tags or ()),
            linked_concepts=tuple(linked_concepts or ()),
        )
        self.notes.append(note)
        return note

    def find_by_tag(self, tag: str) -> list[Note]:
        return sorted((note for note in self.notes if tag in note.tags), key=lambda n: n.title)

    def find_by_concept(self, concept_name: str) -> list[Note]:
        matches = [
            note
            for note in self.notes
            if concept_name.lower() in (item.lower() for item in note.linked_concepts)
        ]
        return sorted(matches, key=lambda n: n.title)


def search(
    query: str,
    graph: KnowledgeGraph,
    notebook: Notebook | None = None,
) -> list[tuple[float, str]]:
    """Ranked keyword hits across concepts, summaries, relations and notes.

    Returns ``(score, description)`` pairs sorted best-first.
    """
    terms = query.lower().split()
    results: list[tuple[float, str]] = []

    for concept in graph.concepts.values():
        haystacks = {"concept": concept.name.lower(), "summary": concept.summary.lower()}
        score = 0.0
        for term in terms:
            if term in haystacks["concept"]:
                score += 2.0
            score += haystacks["summary"].count(term) * 0.5
        if score > 0:
            results.append((score, f"concept:{concept.name}"))

    for relation in graph.relations:
        label_text = relation.label.lower()
        if any(term in label_text for term in terms):
            score = 1.5
            results.append(
                (
                    score,
                    f"relation:{relation.source} -[{relation.label}]-> {relation.target}",
                )
            )

    if notebook is not None:
        for note in notebook.notes:
            body_text = note.body.lower()
            title_text = note.title.lower()
            score = sum(terms.count(term) * 0.2 for term in terms if term in body_text)
            score += sum(2.0 for term in terms if term in title_text.split())
            if score > 0:
                results.append((score, f"note:{note.title}"))

    return sorted(results, key=lambda pair: pair[0], reverse=True)
