"""Find feature definitions that cannot be computed at prediction time.

The package exists because of one blind spot. A cross-validated score cannot expose a
leaking feature pipeline, because the leak lifts every fold together: the training rows
carry their own future and so do the validation rows, so the comparison between them
reveals nothing. The score goes up and every diagnostic stays quiet.

Switching to a time-ordered splitter does not help either, which is the part that surprises
people, because the leak is in the feature's definition rather than in how the rows were
divided. No arrangement of rows fixes a column that was built from the future.

What does work is reading the definitions. Given a window, an ingestion lag and a
dependency graph, whether a value is knowable at the moment of prediction is a question
with an answer, and answering it needs no data and no model.
"""

from leakage_audit.audit import AuditReport, Violation, audit
from leakage_audit.features import FeatureSpec, Pipeline

__all__ = ["AuditReport", "FeatureSpec", "Pipeline", "Violation", "audit"]
__version__ = "0.1.0"
