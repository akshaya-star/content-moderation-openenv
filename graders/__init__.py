"""Deterministic graders for easy, medium, and hard tasks."""

from .easy_grader import grade as grade_easy
from .hard_grader import grade as grade_hard
from .medium_grader import grade as grade_medium

__all__ = ["grade_easy", "grade_medium", "grade_hard"]
