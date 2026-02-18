# Code Execution Lifecycle

> Activate this skill when orchestrating software transformations from source authorship through runtime execution. Use when managing function invocations, object instantiation, compilation pipelines, or operation finalization. Essential for understanding how code transitions between human-readable instructions and machine execution, including control flow transfer, resource lifecycle management, and proper termination patterns. Applies to build systems, runtime environments, API design, and any workflow requiring precise coordination of initialization, processing, and cleanup phases.

## Overview

This skill governs the four fundamental actions that transform static code into computational work: **Call** (invoking functions and transferring control), **Create** (instantiating objects and allocating resources), **Complete** (finalizing operations and returning results), and **Compile** (translating source code to executable binaries). These actions orchestrate the execution lifecycle from initialization through active processing to orderly termination, ensuring proper resource management and semantic preservation across transformation stages.

## When to Use

- Designing function signatures and invocation patterns that require control flow transfer
- Implementing object-oriented systems with constructor/destructor lifecycle management
- Building compilation pipelines that transform source code through lexical, syntactic, and semantic phases
- Ensuring proper operation termination with resource cleanup and result communication
- Debugging runtime issues related to state initialization, memory leaks, or incomplete finalization
- Architecting build systems that coordinate translation, linking, and binary generation
- Managing workflows where timing of creation, execution, and completion critically affects correctness

## Core Workflow

1. **Initialize State**: Allocate resources (memory, handles, connections) and establish valid starting conditions through constructors or initialization functions
2. **Transfer Control**: Invoke functions/methods by establishing stack frames, passing parameters, and redirecting execution flow to target code blocks
3. **Process & Transform**: Execute operations while maintaining state, potentially compiling or translating code through parsing and optimization stages
4. **Finalize & Return**: Complete operations by satisfying postconditions, deallocating resources, and communicating results back to callers
5. **Terminate Cleanly**: Ensure orderly shutdown with proper cleanup, releasing all reserved resources and returning control to appropriate contexts

## Key Patterns

### Function Invocation with Resource Safety

Wrap function calls in context managers or RAII patterns to guarantee resource cleanup even when exceptions occur. Establish clear contracts for parameter passing and return value semantics.

```python
from typing import TypeVar, Callable, Any
from contextlib import contextmanager

T = TypeVar('T')

@contextmanager
def safe_call_context(resource_init: Callable[[], Any]):
    """
    Context manager ensuring resource cleanup after function invocation.
    Allocates resources before call, guarantees deallocation after.
    """
    resource = resource_init()  # Initialize/allocate resource
    try:
        yield resource  # Transfer control to caller's code block
    finally:
        # Termination phase: cleanup regardless of success/exception
        if hasattr(resource, 'close'):
            resource.close()
        elif hasattr(resource, 'cleanup'):
            resource.cleanup()

# Usage: invocation with guaranteed resource lifecycle
def process_data(data: list[int]) -> int:
    """Call target: performs computation with resource dependency."""
    return sum(x * 2 for x in data)

with safe_call_context(lambda: open('data.txt')) as file:
    data = [int(line.strip()) for line in file]
    result = process_data(data)  # Function invocation
    # File resource automatically finalized on context exit
```

### Object Creation with Initialization Protocol

Implement factory patterns and builder protocols to centralize instantiation logic, ensuring all objects begin in valid states with properly allocated resources.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ComputeTask:
    """
    Encapsulates computational work with lifecycle management.
    Creation phase establishes initial state and allocates resources.
    """
    task_id: str
    data: list[int]
    _result: Optional[int] = None
    _completed: bool = False
    
    @classmethod
    def create(cls, task_id: str, data: list[int]) -> 'ComputeTask':
        """
        Factory method: centralized object instantiation with validation.
        Initializes state and allocates necessary structures.
        """
        if not task_id:
            raise ValueError("Task ID required for resource tracking")
        if not data:
            raise ValueError("Cannot create task with empty data")
        
        # Instantiation: allocate memory and establish initial state
        instance = cls(task_id=task_id, data=data.copy())
        print(f"Created task {task_id} with {len(data)} elements")
        return instance
    
    def execute(self) -> int:
        """
        Invocation: perform computation and transition to complete state.
        Control flow transfers to this method, processes data, returns result.
        """
        if self._completed:
            return self._result  # Return cached value if already complete
        
        # Active processing phase
        self._result = sum(x ** 2 for x in self.data)
        self._completed = True  # Operation completion marker
        print(f"Completed task {self.task_id}: result={self._result}")
        return self._result
    
    def __del__(self):
        """
        Termination: cleanup resources during object destruction.
        Ensures proper finalization even if explicit cleanup not called.
        """
        if not self._completed:
            print(f"Warning: Task {self.task_id} destroyed before completion")

# Instantiation and lifecycle demonstration
task = ComputeTask.create("calc_001", [1, 2, 3, 4])  # Create phase
result = task.execute()  # Call phase: invoke computation
# Complete phase: result returned, state finalized
```

### Compilation Pipeline with Stage Validation

Structure code translation as a multi-stage pipeline where each phase validates inputs, transforms representations, and produces artifacts for subsequent stages.

```python
from enum import Enum
from typing import NamedTuple

class CompilationStage(Enum):
    """Stages in the code compilation lifecycle."""
    LEXICAL = "tokenization"
    SYNTACTIC = "parsing"
    SEMANTIC = "validation"
    OPTIMIZATION = "optimization"
    CODE_GEN = "binary_generation"

class CompilationArtifact(NamedTuple):
    """Result of a compilation stage transformation."""
    stage: CompilationStage
    data: str  # Simplified: would contain tokens, AST, IR, or binary
    errors: list[str]
    
    @property
    def is_complete(self) -> bool:
        """Check if this stage completed without errors."""
        return len(self.errors) == 0

class CompilerPipeline:
    """
    Orchestrates code translation through compilation stages.
    Each stage: calls previous stage, creates new representation, completes validation.
    """
    
    def __init__(self, source_code: str):
        self.source = source_code
        self.stages_completed: list[CompilationArtifact] = []
    
    def compile(self) -> CompilationArtifact:
        """
        Execute full compilation lifecycle: lexical -> syntactic -> semantic -> optimize -> generate.
        Returns final binary artifact or error state.
        """
        # Stage 1: Lexical analysis (tokenization)
        tokens = self._tokenize(self.source)
        if not tokens.is_complete:
            return tokens  # Early termination on failure
        
        # Stage 2: Syntactic analysis (parsing)
        ast = self._parse(tokens)
        if not ast.is_complete:
            return ast
        
        # Stage 3: Semantic validation
        validated = self._validate(ast)
        if not validated.is_complete:
            return validated
        
        # Stage 4: Optimization
        optimized = self._optimize(validated)
        
        # Stage 5: Binary generation (final compilation product)
        binary = self._generate_code(optimized)
        return binary  # Complete: final artifact ready
    
    def _tokenize(self, source: str) -> CompilationArtifact:
        """Lexical stage: translate source text into tokens."""
        errors = []
        if not source.strip():
            errors.append("Empty source file")
        
        tokens = source.split()  # Simplified tokenization
        artifact = CompilationArtifact(
            stage=CompilationStage.LEXICAL,
            data=f"TOKENS[{len(tokens)}]",
            errors=errors
        )
        self.stages_completed.append(artifact)
        return artifact
    
    def _parse(self, tokens: CompilationArtifact) -> CompilationArtifact:
        """Syntactic stage: build abstract syntax tree from tokens."""
        artifact = CompilationArtifact(
            stage=CompilationStage.SYNTACTIC,
            data=f"AST from {tokens.data}",
            errors=[]
        )
        self.stages_completed.append(artifact)
        return artifact
    
    def _validate(self, ast: CompilationArtifact) -> CompilationArtifact:
        """Semantic stage: type checking and validation."""
        artifact = CompilationArtifact(
            stage=CompilationStage.SEMANTIC,
            data=f"VALIDATED {ast.data}",
            errors=[]
        )
        self.stages_completed.append(artifact)
        return artifact
    
    def _optimize(self, validated: CompilationArtifact) -> CompilationArtifact:
        """Optimization stage: improve performance while preserving semantics."""
        artifact = CompilationArtifact(
            stage=CompilationStage.OPTIMIZATION,
            data=f"OPTIMIZED {validated.data}",
            errors=[]
        )
        self.stages_completed.append(artifact)
        return artifact
    
    def _generate_code(self, optimized: CompilationArtifact) -> CompilationArtifact:
        """Code generation stage: produce machine-executable binary."""
        artifact = CompilationArtifact(
            stage=CompilationStage.CODE_GEN,
            data=f"BINARY from {optimized.data}",
            errors=[]
        )
        self.stages_completed.append(artifact)
        return artifact

# Compilation lifecycle demonstration
compiler = CompilerPipeline("def main(): return 42")
result = compiler.compile()  # Orchestrates create->call->complete cycle
print(f"Compilation {'complete' if result.is_complete else 'failed'}: {result.stage.value}")
```

### Operation Completion with Return Value Protocol

Establish clear contracts for how operations signal completion, communicate results, and handle error conditions through return values or exception mechanisms.

```python
from typing import Generic, TypeVar, Union
from dataclasses import dataclass

T = TypeVar('T')
E = TypeVar('E', bound=Exception)

@dataclass
class Result(Generic[T, E]):
    """
    Completion protocol: encapsulates operation outcome with value or error.
    Enables explicit handling of both success and failure paths.
    """
    _value: Union[T, None] = None
    _error: Union[E, None] = None
    
    @property
    def is_complete(self) -> bool:
        """Check if operation terminated (successfully or with error)."""
        return self._value is not None or self._error is not None
    
    @property
    def is_success(self) -> bool:
        """Check if operation completed successfully."""
        return self._value is not None and self._error is None
    
    @staticmethod
    def success(value: T) -> 'Result[T, E]':
        """Create completion result for successful operation."""
        return Result(_value=value)
    
    @staticmethod
    def failure(error: E) -> 'Result[T, E]':
        """Create completion result for failed operation."""
        return Result(_error=error)
    
    def unwrap(self) -> T:
        """
        Retrieve successful value or raise error.
        Terminal operation: finalizes result access.
        """
        if self._error:
            raise self._error
        return self._value

def divide_safe(numerator: int, denominator: int) -> Result[float, ValueError]:
    """
    Operation with explicit completion protocol.
    Returns Result indicating success with value or failure with error.
    """
    if denominator == 0:
        return Result.failure(ValueError("Division by zero"))
    
    result = numerator / denominator
    return Result.success(result)  # Complete: return final value

# Usage: explicit completion handling
operation = divide_safe(10, 2)
if operation.is_success:
    value = operation.unwrap()  # Finalize: extract result
    print(f"Operation complete: {value}")
else:
    print("Operation failed to complete")
```

## Concept Reference

| Concept | Technical | Plain | Importance |
|---------|-----------|-------|------------|
| Call | An invocation of a subroutine, function, or method that transfers control flow to a defined code block, typically passing parameters and expecting a result | Asking a specific piece of code to run and do its job, like calling someone on the phone to ask them to perform a task | 0.95 |
| Compile | The process of translating high-level source code into machine-executable binary code or intermediate bytecode through lexical analysis, parsing, semantic validation | Converting human-readable programming instructions into a language that computers can directly understand and run | 0.94 |
| Create | The instantiation or initialization of a new object, data structure, or resource in memory, allocating necessary storage and establishing initial state | Making something new in a program, like creating a new document or setting up a fresh container to hold information | 0.92 |
| Control flow transfer | The redirection of program execution sequence from one instruction location to another, whether through function calls, jumps, branches, or exception handling | Changing which part of the code runs next, like following different paths in a choose-your-own-adventure book | 0.89 |
| Complete | The terminal state of an operation or process indicating successful execution of all required steps, with all postconditions satisfied and resources properly finalized | Finishing a task entirely so that it's done and ready, like completing a form by filling in all the required fields | 0.88 |
| Return value | The data value or object reference passed back from a called function to its caller through the function's return statement, communicating results | The answer or result that a function gives back after it finishes its work, like getting change back after making a purchase | 0.88 |
| Function invocation | The mechanism by which program execution jumps to a function's entry point, establishing a new stack frame with local variables and parameters | The actual moment when you trigger a function to start working, passing it the information it needs to do its job | 0.87 |
| Source code processing | The multi-stage analysis and transformation of textual program code through lexical tokenization, syntactic parsing, semantic validation, and optimization | Reading and understanding written code by breaking it into pieces, checking if it makes sense, and preparing it to run | 0.87 |
| Code translation | The systematic transformation of source code representations between different abstraction levels or target platforms while preserving semantic equivalence | Converting code from one form to another while keeping the meaning and functionality the same, like translating a book between languages | 0.86 |
| Program lifecycle | The complete temporal span from source code authorship through compilation, deployment, execution, and eventual termination or decommission | The entire lifespan of a program from when it's first written until it stops being used, like a product from design to retirement | 0.86 |

## Glossary

| Term | Definition | Concept IDs |
|------|------------|-------------|
| Call | Invoking a function or method to transfer control flow, pass parameters, and await return values | [1, 5, 9] |
| Create | Instantiating new objects or data structures by allocating memory and initializing state | [2, 6, 10, 11] |
| Complete | Reaching the terminal state where operations finish successfully with resources finalized | [3, 12, 14] |
| Compile | Translating source code into machine-executable binaries through parsing and code generation | [4, 8, 13, 15] |
| Invocation | The triggering mechanism that activates functions and establishes execution contexts | [1, 5] |
| Instantiation | Runtime creation of concrete objects from abstract class templates with constructor execution | [2, 6] |
| Lifecycle | The temporal sequence from initialization through active execution to termination | [7, 17] |
| Translation | Converting code between abstraction levels while preserving semantic equivalence | [4, 8] |
| Control Flow | The redirection of execution sequence through calls, branches, and exception handling | [1, 9] |
| Resource Management | Allocation and deallocation of system resources like memory and file handles | [10, 12] |
| Initialization | Establishing valid starting conditions by setting initial values and configurations | [11, 6] |
| Termination | Orderly conclusion of processes with result finalization and control return | [3, 12] |
| Processing Pipeline | Multi-stage analysis transforming source code through tokenization, parsing, and validation | [13, 16, 4] |

## Edge Cases & Warnings

- ⚠️ **Incomplete Finalization**: Functions that allocate resources (files, connections, locks) but lack corresponding cleanup calls create resource leaks. Always pair creation with explicit termination or use RAII/context managers.
- ⚠️ **Premature Optimization in Compilation**: Aggressive optimization during compilation can alter execution semantics or introduce subtle bugs. Preserve debug symbols and validate optimized code against unoptimized baselines.
- ⚠️ **Reentrant Invocation Hazards**: Functions that call themselves or are invoked recursively may exhaust stack space or violate state consistency assumptions. Guard recursive calls with depth limits and ensure idempotent operations.
- ⚠️ **Initialization Order Dependencies**: Objects created in specific sequences may have implicit dependencies. Document and enforce initialization order or use lazy initialization patterns to avoid undefined behavior.
- ⚠️ **Return Value Ignoring**: Callers that ignore return values or error codes miss critical completion signals. Use Result types, exceptions, or linting tools to enforce error handling discipline.
- ⚠️ **Partial Compilation Artifacts**: Build systems that cache intermediate compilation stages may serve stale artifacts after source changes. Implement dependency tracking and checksums to trigger recompilation when needed.
- ⚠️ **Non-Deterministic Termination**: Operations with unbounded execution time (infinite loops, blocking I/O) may never complete. Implement timeouts, cancellation tokens, and watchdog timers for critical workflows.

## Quick Reference

```python
from typing import TypeVar, Callable
from contextlib import contextmanager

T = TypeVar('T')

# CALL: Invoke with guaranteed cleanup
@contextmanager
def safe_invoke(setup: Callable[[], T]):
    resource = setup()  # CREATE: allocate
    try:
        yield resource  # Transfer control
    finally:
        # COMPLETE: finalize regardless of outcome
        if hasattr(resource, 'close'):
            resource.close()

# CREATE: Factory pattern for initialization
class Task:
    @classmethod
    def create(cls, data: list) -> 'Task':
        return cls(data.copy())  # Initialize with validated state

# COMPLETE: Explicit result protocol
def compute(x: int) -> tuple[bool, int]:
    """Returns (success, result) completion indicator."""
    if x < 0:
        return (False, 0)  # Failed completion
    return (True, x * 2)  # Successful completion

# COMPILE: Multi-stage transformation
def compile_pipeline(source: str) -> bytes:
    tokens = tokenize(source)      # Stage 1: lexical
    ast = parse(tokens)            # Stage 2: syntactic
    validated = validate(ast)      # Stage 3: semantic
    return generate_binary(validated)  # Stage 4: code gen
```

---
_Generated by Philosopher's Stone v4 — EchoSeed_
Philosopher's Stone v4 × Skill Forge × EchoSeed
