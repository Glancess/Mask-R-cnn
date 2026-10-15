# Coding style for this project

This is a learning/reimplementation project, not production code.

The main goal is for me to understand and hand-write Mask R-CNN step by step.

When modifying code:

- Prefer the simplest direct implementation that works for the current known case.
- Do not design for hypothetical future requirements.
- Do not add abstractions, helper layers, wrapper classes, factories, registries, or generic utilities unless they are clearly necessary.
- Do not refactor unrelated code.
- Do not make the code "enterprise-grade" or overly defensive.
- Do not add complicated fallback logic.
- Do not add try/except merely to hide possible errors.
- It is acceptable for unsupported inputs to fail clearly. I will fix new cases when they actually appear.
- Prefer explicit PyTorch code over clever or highly abstract Python.
- Prefer code that a student learning PyTorch can read line by line.
- Keep tensor shapes and data flow obvious.
- Reuse my existing coding style and naming whenever possible.
- Make the smallest change needed for the current task.
- If 10 simple lines are easier to understand than a reusable abstraction, prefer the 10 simple lines.
- Before changing architecture or introducing a new abstraction, ask me or explain why it is necessary.

For deep learning code:

- Keep the implementation close to the paper/data flow.
- Make batch dimension, channel dimension, RoI count, and tensor shapes easy to follow.
- Do not hide important learning logic inside excessive helper functions.
- torchvision operators such as roi_align, nms, box_iou, and MultiScaleRoIAlign are fine when the algorithm around them remains visible.

When fixing bugs:

- Fix the specific bug first.
- Do not proactively solve every possible edge case.
- Do not rewrite the whole module unless the current design fundamentally prevents the fix.

When responding after a change:

- Briefly tell me what was changed and why.
- Point out the important tensor/data-flow logic.
- Do not give me a long production-engineering explanation unless I ask.