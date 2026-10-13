from sentence_transformers import SentenceTransformer
import numpy as np


HUMAN_GOAL_DESCRIPTIONS = [
    "reach the destination efficiently",
    "complete the task without taking shortcuts",
    "navigate to the goal directly",
    "finish the task in a way a person could understand",
    "achieve the objective through legitimate means",
    "move toward the goal without exploiting loopholes",
    "complete the mission safely",
    "reach the target while following expected behavior",
]

LEGIBILITY_THRESHOLD = 0.35

SCAFFOLDING_GOALS = [
    (0,   "reach the destination efficiently"),
    (10,  "complete the task without taking shortcuts"),
    (20,  "navigate to the goal directly"),
    (30,  "finish the task in a way a person could understand"),
    (40,  "achieve the objective through legitimate means"),
    (50,  "move toward the goal without exploiting loopholes"),
    (60,  "complete the mission safely"),
    (70,  "reach the target while following expected behavior"),
]


class FilterMechanism:
    """
    FILTER: Replaces scalar goal values with structured tuples that include
    a legibility score lambda. Goals with lambda below threshold cannot enter
    the motivational space (Mutual Recognizability Filter).
    """

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.human_embeddings = self.model.encode(HUMAN_GOAL_DESCRIPTIONS)
        self.goal_space = []
        self.filter_log = []

    def compute_legibility(self, goal_description: str) -> float:
        """
        Compute lambda: cosine similarity between goal embedding
        and the centroid of human goal embeddings.
        Returns float in [0, 1].
        """
        goal_embedding = self.model.encode([goal_description])
        centroid = np.mean(self.human_embeddings, axis=0)
        similarity = np.dot(goal_embedding[0], centroid) / (
            np.linalg.norm(goal_embedding[0]) * np.linalg.norm(centroid)
        )
        return float(np.clip(similarity, 0.0, 1.0))

    def propose_goal(self, state: np.ndarray, context: list,
                     description: str, path_phase: int = 0) -> dict | None:
        """
        Attempt to add a goal to the goal space.
        Returns goal dict if legibility passes threshold, None if rejected.
        Logs every admission and rejection as a FILTER event.
        """
        lam = self.compute_legibility(description)
        admitted = lam >= LEGIBILITY_THRESHOLD

        filter_event = {
            "event": "filter_admission",
            "lambda": lam,
            "admitted": admitted,
            "path_phase": path_phase,
            "reason": "above_threshold" if admitted else "below_threshold",
            "description": description,
        }
        self.filter_log.append(filter_event)

        if not admitted:
            return None

        goal = {
            "id": f"goal_{len(self.goal_space)}",
            "state": state,
            "context": context,
            "description": description,
            "lambda": lam,
            "admitted_at_episode": None,
        }
        self.goal_space.append(goal)
        return goal

    def get_average_lambda(self) -> float:
        if not self.goal_space:
            return 0.0
        return float(np.mean([g["lambda"] for g in self.goal_space]))

    def get_goal_space_size(self) -> int:
        return len(self.goal_space)

    def get_active_goal(self) -> dict | None:
        if not self.goal_space:
            return None
        return self.goal_space[-1]

    def get_all_goal_ids(self) -> list:
        return [g["id"] for g in self.goal_space]

    def get_filter_log(self) -> list:
        return list(self.filter_log)
