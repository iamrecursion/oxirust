# SciRS2 Dependency Strategy

## Overview

This document describes the strategy for managing dependencies between modules in the SciRS2 project. Proper dependency management is essential to achieve the following objectives:

- Promote code reuse and prevent reinventing the wheel
- Prevent circular dependencies
- Improve maintainability and extensibility
- Minimize duplication of functionality

## Core Principles

1. **Hierarchical Dependencies**: Modules follow a clear hierarchy in their dependencies
2. **Minimal Dependencies**: Modules should have only the minimum necessary dependencies
3. **Dependency Direction**: Higher-level modules depend on lower-level modules, not vice versa
4. **Shared Dependencies**: Functionality needed by multiple modules should be placed in appropriate shared modules

## Dependency Hierarchy

The dependency hierarchy for SciRS2 is as follows:

1. **Core Layer**: `scirs2-core`
   - Basic utilities and constants that don't depend on other modules

2. **Basic Scientific Computing Layer**: 
   - `scirs2-linalg`, `scirs2-stats`, `scirs2-special`, etc.
   - Depend only on the core layer

3. **Applied Scientific Computing Layer**:
   - `scirs2-optimize`, `scirs2-integrate`, `scirs2-interpolate`, etc.
   - May depend on the core layer and basic scientific computing layer

4. **Advanced Modules Layer**:
   - `scirs2-cluster`, `scirs2-ndimage`, `scirs2-io`, `scirs2-datasets`
   - May depend on all lower layers

5. **AI/ML Layer**:
   - `scirs2-neural`, `scirs2-optim`, `scirs2-graph`, etc.
   - May depend on all lower layers

## Module Dependency Map

### Basic Scientific Computing Modules

| Module | Dependencies |
|--------|--------------|
| scirs2-core | None |
| scirs2-linalg | scirs2-core |
| scirs2-stats | scirs2-core |
| scirs2-special | scirs2-core |
| scirs2-fft | scirs2-core |
| scirs2-signal | scirs2-core, scirs2-fft |
| scirs2-sparse | scirs2-core, scirs2-linalg |
| scirs2-spatial | scirs2-core |

### Applied Scientific Computing Modules

| Module | Dependencies |
|--------|--------------|
| scirs2-optimize | scirs2-core, scirs2-linalg |
| scirs2-integrate | scirs2-core, scirs2-linalg |
| scirs2-interpolate | scirs2-core, scirs2-linalg |

### Advanced Modules

| Module | Dependencies |
|--------|--------------|
| scirs2-cluster | scirs2-core, scirs2-linalg, scirs2-stats |
| scirs2-ndimage | scirs2-core, scirs2-interpolate |
| scirs2-io | scirs2-core |
| scirs2-datasets | scirs2-core, scirs2-stats |

### AI/ML Modules

| Module | Dependencies |
|--------|--------------|
| scirs2-neural | scirs2-core, scirs2-linalg, scirs2-optimize, scirs2-optim |
| scirs2-optim | scirs2-core, scirs2-linalg, scirs2-optimize |
| scirs2-graph | scirs2-core, scirs2-linalg, scirs2-sparse |
| scirs2-transform | scirs2-core, scirs2-linalg, scirs2-stats |
| scirs2-metrics | scirs2-core, scirs2-stats |
| scirs2-text | scirs2-core, scirs2-linalg |
| scirs2-vision | scirs2-core, scirs2-ndimage |
| scirs2-series | scirs2-core, scirs2-stats, scirs2-fft |

## Functionality Placement Guidelines

When adding new functionality, follow these guidelines to choose the appropriate module:

1. **Generality**: If functionality might be used by multiple modules, place it in a lower-layer module if possible
2. **Specialization**: Functionality specific to a particular domain should be placed in a module dedicated to that domain
3. **Dependencies**: If new functionality depends on other modules, place it in the most appropriate module that already has those dependencies
4. **Consistency**: Place new functionality in the same module as similar existing functionality

## Guidelines for Adding Dependencies

When adding new dependencies, follow these guidelines:

1. **Verify Necessity**: Consider whether the dependency is truly necessary
2. **Minimal Usage**: Use only the minimum necessary functionality from the dependency
3. **Consider Direction**: Ensure the dependency direction follows hierarchical principles to avoid circular dependencies
4. **Evaluate Impact**: Assess the impact of adding the dependency on build time and package size

## Special Considerations for AI/ML Modules

AI/ML modules require special considerations compared to other modules:

1. **Interdependencies**: Interdependencies between AI/ML modules are allowed but should be minimized
2. **Feedback Mechanism**: For functionality needed in AI/ML modules but not present in basic modules, a feedback mechanism should be in place to add them to basic modules
3. **External Dependencies**: Dependencies on external libraries (PandRS, NumRS, etc.) should be handled through appropriate abstractions
4. **Prototyping**: Experimental implementations of new algorithms should be done within AI/ML modules and moved to appropriate layers once stabilized

## Feedback Process

Process for requesting functionality additions to basic modules:

1. **Document Requirements**: Document detailed requirements for the needed functionality
2. **Present Use Cases**: Present concrete examples of how the functionality will be used
3. **Propose Implementation**: If possible, propose an implementation
4. **Review Process**: Review by core developers
5. **Merge Process**: After approval, implement in the appropriate module

## Dependency Update Process

When module dependencies need to be updated:

1. **Propose Update**: Document the necessity and reasons for the dependency update
2. **Impact Analysis**: Analyze the scope of impact
3. **Migration Plan**: Create a plan for migrating existing code
4. **Test Plan**: Plan how to verify the update
5. **Incremental Implementation**: Implement changes in small, incremental steps

## Recommended Practices

1. **Regular Dependency Graph Reviews**: Regularly review the project's overall dependency graph to identify optimization opportunities
2. **Dependency Annotations**: Document reasons for dependencies in code comments
3. **Consider Alternative Designs**: When adding new functionality, consider alternative designs that minimize dependencies
4. **Document Common Patterns**: Document patterns for where specific types of functionality should be placed

## Example Design Decisions

This section records examples of important design decisions related to dependencies:

1. **scirs2-optim vs scirs2-optimize**: 
   - `scirs2-optimize` provides general optimization algorithms
   - `scirs2-optim` provides AI/ML-specific optimization algorithms (e.g., stochastic gradient descent)

2. **ndimage Migration**:
   - The `ndimage` module was moved from the main `scirs2` crate to an independent module
   - Reason: Image processing functionality is likely to be used independently of other modules

3. **PandRS and NumRS Integration**:
   - Implement wrapper interfaces for using PandRS and NumRS in AI/ML modules
   - Reason: To ensure flexibility for future changes