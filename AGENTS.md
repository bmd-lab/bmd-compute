# AGENTS.md

# bmd-compute

This document defines the architectural philosophy of bmd-compute.

It exists to keep human developers and coding agents aligned as the project evolves.

This document describes **how decisions should be made**, not merely how the project is currently implemented.

---

# Project Goal

bmd-compute is a browser-based interface for building, submitting, monitoring and retrieving VASP calculations using an Atomate2 / Jobflow / PowerSLURM backend.

The long-term goal is for bmd-compute to replace the existing Jupyter notebook as the primary user interface.

The notebook is retained as historical/reference scientific material and validation context. It is not a competing live implementation authority for current bmd-compute behavior.

---

# Primary Design Principle

The browser interface is the product.

Everything else exists to support it.

Users should interact with bmd-compute through a browser rather than notebooks, SSH sessions or shell scripts.

---

# The Notebook

The notebook located in `reference/` is **not legacy code**.

It represents important validated scientific workflow history and reference evidence.

Its scientific behaviour should be preserved wherever practical.

When migrating notebook functionality:

Notebook feature

↓

Reusable backend module

↓

Web interface

Do **not** rewrite notebook functionality directly inside FastAPI.

Refactor behaviour into reusable backend modules.

Current backend modules and producer contracts define currently implemented bmd-compute behavior. The notebook remains important historical/reference material to compare against when migrating or revising behavior, but it should not be treated as a separate live authority once behavior is implemented in bmd-compute.

---

# Architecture

The intended architecture is

Browser

↓

FastAPI

↓

Backend

↓

Atomate2

↓

Jobflow

↓

PowerSLURM

↓

VASP

FastAPI should remain a thin controller.

Scientific logic belongs in the backend.

---

# Responsibilities

## FastAPI

FastAPI should only

* receive requests
* validate inputs
* call backend functions
* return responses

FastAPI should **not** contain

* workflow construction
* pymatgen logic
* Atomate2 logic
* SLURM logic
* scientific calculations

---

## Backend

The backend owns all scientific behaviour.

Typical backend modules include

* structures
* workflows
* submission
* monitoring
* results

Backend modules should remain usable independently of the web interface.

---

## Infrastructure

Infrastructure code is separate from scientific code.

Examples include

* SSH
* SLURM
* remote execution
* configuration
* deployment

Scientific modules should not depend directly on infrastructure wherever practical.

---

# Structure Philosophy

Every supported input method

* POSCAR
* CIF
* uploaded file
* Materials Project
* future databases

should converge as early as possible to

`pymatgen.Structure`

Once a Structure exists, downstream workflow code should not care where it originated.

---

# Scientific Workflows

Expose scientific workflows rather than implementation details.

Preferred interface

* Relaxation
* Static
* Band Structure
* HSE06

Avoid exposing large numbers of INCAR parameters unless there is a compelling scientific reason.

Users should think in terms of scientific tasks rather than VASP implementation details.

---

# Stable Interfaces

Backend modules should expose stable interfaces.

Implementation may evolve.

Interfaces should change only when necessary.

For example,

`parse_structure()`

may become more sophisticated internally while remaining the canonical interface for structure parsing.

---

# Development Philosophy

Develop incrementally.

Every commit should

* run
* be testable
* leave the application in a working state

Avoid large rewrites.

Prefer many small working commits over one large implementation.

---

# Deployment Philosophy

Local development should closely mirror the deployed application.

Avoid implementing features that only function in the local development environment.

The deployed application should use the same codebase as local development.

---

# Application State

Avoid relying on temporary local files as the application grows.

Eventually, application state should be represented explicitly through objects such as

* WorkflowSpec
* JobRecord
* ResultSummary

rather than scattered JSON files.

This should happen naturally as functionality is added rather than through premature abstraction.

---

# Migration Strategy

When migrating notebook functionality:

1. Understand the notebook behaviour.
2. Extract reusable backend functionality.
3. Test backend functionality independently.
4. Connect it to FastAPI.
5. Replace the notebook feature.

Never migrate notebook code directly into FastAPI.

---

# Repository Philosophy

The repository should remain understandable to a new developer.

Prefer

```
backend/
```

over large monolithic modules.

Keep scientific code, infrastructure code and web code clearly separated.

---

# Long-Term Vision

A user should be able to

1. Enter a structure.
2. Choose a workflow.
3. Submit.
4. Monitor.
5. Retrieve results.

without directly interacting with

* SSH
* SLURM
* Atomate2 internals
* Jobflow internals

The computational infrastructure should remain hidden behind a clean scientific interface.

---

# Design Rule

Whenever making an architectural decision, ask:

> Does this move bmd-compute closer to replacing the notebook?

If the answer is no, reconsider the design.

The objective is not merely to build a web interface.

The objective is to build a maintainable scientific application that eventually renders the notebook unnecessary while preserving its validated scientific workflows.

The reference notebook is treated as a source of validated scientific context and historical behavior. Current backend modules and producer contracts are the authority for currently implemented bmd-compute behavior. Backend modules should be extracted from or compared with the notebook in small, reviewable units rather than rewritten wholesale.

Notebook migration should occur one backend module at a time. Each extracted module should be independently testable before integration into the web interface.

## User Experience Philosophy

bmd-compute should behave like a scientific workbench rather than a traditional website.

The preferred interaction model is progressive disclosure:

Structure
    ↓
Structure Summary
    ↓
Workflow
    ↓
Workflow Summary
    ↓
Submission
    ↓
Monitoring
    ↓
Results

Each successful action reveals the next stage while preserving all previous information.

Avoid unnecessary page navigation.
