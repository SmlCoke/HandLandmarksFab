# AGENTS.md

## I. Purpose

This file defines the repository-wide instructions for coding agents.

Agents must follow these instructions when inspecting, modifying, testing, or
documenting this repository.

The primary goals are:

1. Preserve repository correctness and maintainability.
2. Make only changes required by the current task.
3. Avoid destructive, unrelated, or speculative modifications.

---

## II. Instruction Priority

Follow instructions in this order:

1. Explicit instructions in the current user request.
2. The nearest applicable `AGENTS.override.md`.
3. The nearest applicable `AGENTS.md`.
4. This repository-level `AGENTS.md`.
5. Existing project conventions inferred from nearby code.

Higher-priority instructions override lower-priority instructions.

Do not interpret a previous task's temporary authorization as authorization for
the current task.

When two applicable instructions conflict and the conflict cannot be resolved
safely, stop before modifying files and explain the conflict.

---

## III. Repository Overview

### 3.1 Project purpose

We are developing a real-time sign language recognition system designed for low-compute (0.8 TOPS) edge devices. The system comprises a pipeline of three models operating in series:
1. Palm Detector: Processes images captured by the camera and outputs palm bounding box coordinates along with the coordinates of two auxiliary points.
2. Hand Landmarker: Performs inference on the Hand ROI defined by the palm bounding box to determine the coordinates of 21 skeletal keypoints, as well as hand presence and handedness (left/right) confidence scores.
3. Gloss Translator: An isolated sign classification model that utilizes the outputs from the Palm Detector and Hand Landmarker; it maintains a temporal window of a specific duration and performs temporal modeling to output the sign language gloss corresponding to the action performed during that interval.

This repository contains the dataset generation system for the Hand Landmarker model, serving as the upstream component for the training system (HLML).

### 3.2 Entry-point documents

- `docs\annotating_system\HLMF_annotating_workflow.md`: Referred to as the "workflow" document; it explains the current system's workflow and procedures rather than serving merely as an operational manual. 
- `docs\annotating_system\HLMF_quick_start.md`: Referred to as the "quick_start" document; a simplified version of the "workflow" document containing instructions for executing the full process, designed for getting started quickly.
- `docs\annotating_system\HLMF_current_status.md`: Referred to as the "current_status" document; it records the current annotation status of the dataset, reflecting the latest version's status.
- `docs\annotating_system\HLMF_data_contract.md`: Referred to as the "data_contract" document; it defines the data directories and data interfaces produced by the current system.

## IV. General Working Rules

### 4.1 Docs Modifying Rules

- The "workflow" document records only the commands, content, and underlying principles for each operational step of the system; it is independent of the system's historical state, the training status of models on the server, the state of the server-side data warehouse, and the project's future plans. It is necessary to explain the command and input (including directory locations) for each step, the actions performed, the output (including directory locations), and the rationale behind parameter adjustments in the YAML configuration file. Please keep this principle in mind when making modifications.
- The "quick_start" document is a simplified version of the "workflow" document; it contains only the commands for each operational step and omits explanations of the underlying principles. Include the name of the process stage for each step and briefly describe the inputs and outputs. Please keep this principle in mind when making modifications. 
- The "current_status" document records the current state of the system and the server-side data warehouse. Please keep this principle in mind when making modifications.
- The "data_contract" document records the data directories and data interfaces produced by the current system. Please keep this principle in mind when making modifications.

These four documents have distinct roles and independent content; each should avoid extensive discussion of the material covered in the others.
These four documents serve as the primary interface documentation for the current repository and are critical; they must be kept synchronized whenever there are subsequent updates to code, configurations, or other documentation.

### 4.2 Principle of simplification

1. Whether during manual operations or automated execution of the repository's programs, performing a hash check (SHA256) at every step is prohibited, as this results in significant waste of time and excessive disk space usage. Datasets from different sources can be effectively isolated based on details such as their source names.
2. During each operation, the agent should perform only the tasks explicitly required by the prompt; maintaining simplicity avoids unnecessary, redundant auditing and verification, which would otherwise waste time.
3. Whenever this repository is updated, there is no need to maintain compatibility with legacy modes, commands, or data assets unless explicitly requested by the user.

### 4.3 Principle of `HAND_DATASET_ROOT`

1. `HAND_DATASET_ROOT` serves as the directory for the persistent storage of training datasets, and all annotation activities within the system are conducted here. An annotated dataset must not be tied to a specific training run ID; instead, it should be reusable and compatible with multiple repositories. Therefore, when modifying this repository, you are prohibited from embedding markers such as "training batch" or "training run ID" into the annotation workflow.