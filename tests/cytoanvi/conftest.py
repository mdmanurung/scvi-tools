"""Shared fixtures and helpers for the cytoanvi test suite."""


class MockTreeNode:
    """Minimal stand-in for scHPL TreeNode — no scHPL install required.

    ``name`` is stored as a list (matching scHPL's convention); pass a string or list.
    """

    def __init__(self, name, descendants=None):
        self.name = name if isinstance(name, (list, tuple)) else [name]
        self.descendants = list(descendants or [])
        self.ancestor = None
        for child in self.descendants:
            child.ancestor = self

    def get_leaves(self):
        if not self.descendants:
            return [self]
        leaves = []
        for child in self.descendants:
            leaves.extend(child.get_leaves())
        return leaves


class ScalarNameTreeNode(MockTreeNode):
    """Mock scHPL node variant whose ``name`` is a plain string (not a list)."""

    def __init__(self, name, descendants=None):
        self.name = name
        self.descendants = list(descendants or [])
        self.ancestor = None
        for child in self.descendants:
            child.ancestor = self
