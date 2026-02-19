# Multi-Agent Project Orchestration

> Trigger this skill when managing autonomous computational entities executing concurrent workloads across isolated project contexts. Essential for distributed systems requiring coordinated parallel execution, dynamic resource allocation, and fault-tolerant operation at scale. Apply when coordination complexity, project isolation requirements, or agent population size exceeds manual management capacity. Critical for maintaining system observability, preventing resource contention, and ensuring graceful degradation under partial failures across heterogeneous multi-project environments.

## Overview

This skill provides orchestration frameworks for managing fleets of parallel agents distributed across multiple isolated project contexts. It addresses the coordination challenges inherent in distributed autonomous systems: maintaining execution independence while enabling strategic cooperation, allocating shared computational resources fairly across competing workloads, tracking distributed state across network partitions, and ensuring operational continuity despite component failures. The skill encompasses lifecycle management from agent initialization through graceful termination, inter-agent communication protocols, context isolation enforcement, and comprehensive observability infrastructure for runtime introspection at scale.

## When to Use

- Managing 3+ autonomous agents executing concurrent workloads across system boundaries
- Coordinating work across 2+ isolated project contexts with separate resource allocations
- System requires fault tolerance with graceful degradation under partial agent failures
- Need dynamic resource allocation and load balancing across heterogeneous agent capabilities
- Distributed execution requires state synchronization and inter-agent communication
- Observability demands exceed local logging; require distributed tracing and aggregated metrics
- Scalability requirements necessitate horizontal scaling through agent instance multiplication

## Core Workflow

1. **Initialization Phase**: Configure orchestration layer with project boundaries, resource quotas, and agent specifications. Define isolation policies, communication channels, and supervision hierarchies. Establish monitoring infrastructure with health check endpoints and metric collection.

2. **Agent Provisioning**: Instantiate agent instances according to workload specifications. Assign project context, inject environment-specific configuration, and register with orchestration layer. Establish communication channels and initialize state tracking mechanisms.

3. **Resource Allocation**: Execute scheduling algorithm to distribute computational resources (CPU, memory, I/O bandwidth) across active agents. Apply fairness constraints, priority weights, and project-level quotas. Continuously rebalance based on runtime metrics and changing workload characteristics.

4. **Execution Coordination**: Orchestrate agent interactions through message-passing protocols. Enforce dependency constraints, manage task queues, and coordinate cross-agent workflows. Maintain context isolation between project boundaries while enabling controlled inter-project communication.

5. **State Management**: Track agent execution state, context variables, and progress indicators. Synchronize state across distributed nodes, handle persistence requirements, and enable checkpoint/restart capabilities for fault recovery.

6. **Health Monitoring**: Collect runtime metrics, distributed traces, and log streams from all agents. Detect anomalies, trigger alerts on threshold violations, and provide real-time visibility into system health and performance characteristics.

7. **Failure Recovery**: Detect agent failures through missed heartbeats or explicit error signals. Execute supervision strategies (restart, escalate, isolate). Redistribute workload from failed agents, maintain service continuity, and prevent cascading failures.

8. **Graceful Shutdown**: Coordinate orderly agent termination. Complete in-flight work, persist state, release resources, and deregister from orchestration layer. Execute cleanup procedures and generate shutdown audit trails.

## Key Patterns

### Agent Pool Management

Maintain dynamic pools of agent instances per project context, scaling population based on workload demand while respecting resource constraints and isolation boundaries.

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol
from enum import Enum
import asyncio
from datetime import datetime


class AgentState(Enum):
    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"
    TERMINATING = "terminating"


@dataclass
class ProjectContext:
    """Isolation boundary for multi-project execution."""
    project_id: str
    resource_quota: Dict[str, float]  # e.g., {"cpu": 2.0, "memory_gb": 4.0}
    max_agents: int
    isolation_namespace: str


@dataclass
class Agent:
    """Individual autonomous agent with lifecycle tracking."""
    agent_id: str
    project_id: str
    state: AgentState = AgentState.INITIALIZING
    current_task: Optional[str] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)
    last_heartbeat: Optional[datetime] = None
    
    async def execute_task(self, task: str) -> None:
        """Execute assigned task with state transitions."""
        self.state = AgentState.BUSY
        self.current_task = task
        # Task execution logic here
        await asyncio.sleep(0.1)  # Simulate work
        self.state = AgentState.IDLE
        self.current_task = None


class AgentPool:
    """Dynamic pool of agents with scaling and lifecycle management."""
    
    def __init__(self, context: ProjectContext):
        self.context = context
        self.agents: Dict[str, Agent] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        
    async def scale_to(self, target_size: int) -> None:
        """Scale pool to target size respecting quota constraints."""
        target_size = min(target_size, self.context.max_agents)
        
        # Scale up
        while len(self.agents) < target_size:
            agent = Agent(
                agent_id=f"{self.context.project_id}_agent_{len(self.agents)}",
                project_id=self.context.project_id
            )
            self.agents[agent.agent_id] = agent
            agent.state = AgentState.IDLE
            
        # Scale down
        while len(self.agents) > target_size:
            # Find idle agent to terminate
            idle_agents = [a for a in self.agents.values() 
                          if a.state == AgentState.IDLE]
            if idle_agents:
                agent = idle_agents[0]
                agent.state = AgentState.TERMINATING
                del self.agents[agent.agent_id]
    
    def get_available_agent(self) -> Optional[Agent]:
        """Retrieve idle agent for task assignment."""
        for agent in self.agents.values():
            if agent.state == AgentState.IDLE:
                return agent
        return None
    
    async def submit_task(self, task: str) -> None:
        """Queue task for execution by available agent."""
        await self.task_queue.put(task)
    
    async def process_tasks(self) -> None:
        """Worker loop processing queued tasks with available agents."""
        while True:
            task = await self.task_queue.get()
            agent = self.get_available_agent()
            
            if agent is None:
                # No agents available, requeue
                await self.task_queue.put(task)
                await asyncio.sleep(0.1)
                continue
                
            await agent.execute_task(task)
            self.task_queue.task_done()
```

### Multi-Project Orchestrator

Centralized coordinator managing agent pools across isolated project contexts with resource allocation, load balancing, and cross-project visibility.

```python
from collections import defaultdict
from typing import List, Dict
import logging


@dataclass
class ResourceAllocation:
    """Current resource distribution across projects."""
    project_id: str
    allocated_cpu: float
    allocated_memory_gb: float
    active_agents: int
    pending_tasks: int


class MultiProjectOrchestrator:
    """Central orchestration layer for multi-project agent management."""
    
    def __init__(self, total_resources: Dict[str, float]):
        self.total_resources = total_resources
        self.projects: Dict[str, AgentPool] = {}
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        self.logger = logging.getLogger(__name__)
        
    def register_project(self, context: ProjectContext) -> AgentPool:
        """Register new project context with isolated agent pool."""
        if context.project_id in self.projects:
            raise ValueError(f"Project {context.project_id} already registered")
        
        # Validate resource quota against total capacity
        for resource, quota in context.resource_quota.items():
            allocated = sum(
                alloc.allocated_cpu if resource == "cpu" 
                else alloc.allocated_memory_gb
                for alloc in self.resource_allocations.values()
            )
            if allocated + quota > self.total_resources.get(resource, 0):
                raise ValueError(f"Insufficient {resource} for project quota")
        
        pool = AgentPool(context)
        self.projects[context.project_id] = pool
        
        self.resource_allocations[context.project_id] = ResourceAllocation(
            project_id=context.project_id,
            allocated_cpu=0.0,
            allocated_memory_gb=0.0,
            active_agents=0,
            pending_tasks=0
        )
        
        self.logger.info(f"Registered project {context.project_id} "
                        f"with quota {context.resource_quota}")
        return pool
    
    async def distribute_workload(self) -> None:
        """Load balancing across projects based on queue depth."""
        for project_id, pool in self.projects.items():
            allocation = self.resource_allocations[project_id]
            
            # Calculate target agent count based on queue depth
            queue_depth = pool.task_queue.qsize()
            allocation.pending_tasks = queue_depth
            
            # Simple scaling policy: 1 agent per 5 queued tasks, min 1
            target_agents = max(1, queue_depth // 5)
            target_agents = min(target_agents, pool.context.max_agents)
            
            await pool.scale_to(target_agents)
            allocation.active_agents = len(pool.agents)
    
    def get_system_status(self) -> List[ResourceAllocation]:
        """Retrieve current resource allocation across all projects."""
        return list(self.resource_allocations.values())
    
    async def submit_task_to_project(self, project_id: str, task: str) -> None:
        """Route task to specific project's agent pool."""
        if project_id not in self.projects:
            raise ValueError(f"Unknown project: {project_id}")
        
        pool = self.projects[project_id]
        await pool.submit_task(task)
        
        # Trigger load balancing on new work submission
        await self.distribute_workload()
```

### State Synchronization Protocol

Distributed state management ensuring consistency across agent instances with eventual consistency guarantees and conflict resolution.

```python
from typing import Any, Callable, Optional
from datetime import datetime, timedelta
import json


@dataclass
class StateVector:
    """Versioned state snapshot with causality tracking."""
    agent_id: str
    version: int
    timestamp: datetime
    data: Dict[str, Any]
    checksum: str
    
    def compute_checksum(self) -> str:
        """Generate deterministic checksum for conflict detection."""
        serialized = json.dumps(self.data, sort_keys=True)
        return str(hash(serialized))


class StateStore(Protocol):
    """Abstract state persistence interface."""
    async def read(self, agent_id: str) -> Optional[StateVector]: ...
    async def write(self, state: StateVector) -> None: ...


class InMemoryStateStore:
    """Simple in-memory implementation of StateStore."""
    def __init__(self):
        self.states: Dict[str, StateVector] = {}
    
    async def read(self, agent_id: str) -> Optional[StateVector]:
        return self.states.get(agent_id)
    
    async def write(self, state: StateVector) -> None:
        self.states[agent_id] = state


class StateSynchronizer:
    """Manages state synchronization with conflict resolution."""
    
    def __init__(self, store: StateStore, sync_interval: timedelta):
        self.store = store
        self.sync_interval = sync_interval
        self.local_states: Dict[str, StateVector] = {}
        self.conflict_resolver: Optional[Callable] = None
        
    def set_conflict_resolver(self, 
                            resolver: Callable[[StateVector, StateVector], 
                                             StateVector]) -> None:
        """Register custom conflict resolution strategy."""
        self.conflict_resolver = resolver
    
    async def update_local_state(self, agent_id: str, 
                                  updates: Dict[str, Any]) -> None:
        """Apply updates to local state vector."""
        current = self.local_states.get(agent_id)
        
        if current is None:
            # Initialize new state vector
            current = StateVector(
                agent_id=agent_id,
                version=1,
                timestamp=datetime.utcnow(),
                data=updates,
                checksum=""
            )
        else:
            # Increment version and merge updates
            current = StateVector(
                agent_id=agent_id,
                version=current.version + 1,
                timestamp=datetime.utcnow(),
                data={**current.data, **updates},
                checksum=""
            )
        
        current.checksum = current.compute_checksum()
        self.local_states[agent_id] = current
    
    async def sync(self, agent_id: str) -> None:
        """Synchronize local state with distributed store."""
        local = self.local_states.get(agent_id)
        if local is None:
            return
        
        # Read remote state
        remote = await self.store.read(agent_id)
        
        if remote is None:
            # No remote state, write local
            await self.store.write(local)
            return
        
        # Conflict detection
        if remote.version > local.version:
            # Remote is newer, check for conflicts
            if remote.checksum != local.checksum:
                # Conflict detected, resolve
                resolved = await self._resolve_conflict(local, remote)
                self.local_states[agent_id] = resolved
                await self.store.write(resolved)
            else:
                # No conflict, adopt remote
                self.local_states[agent_id] = remote
        elif local.version > remote.version:
            # Local is newer, push to remote
            await self.store.write(local)
    
    async def _resolve_conflict(self, local: StateVector, 
                               remote: StateVector) -> StateVector:
        """Apply conflict resolution strategy."""
        if self.conflict_resolver:
            return self.conflict_resolver(local, remote)
        
        # Default: last-write-wins based on timestamp
        winner = local if local.timestamp > remote.timestamp else remote
        return StateVector(
            agent_id=winner.agent_id,
            version=max(local.version, remote.version) + 1,
            timestamp=datetime.utcnow(),
            data=winner.data,
            checksum=winner.checksum
        )
```

### Fault Tolerance Supervisor

Supervision tree implementing hierarchical failure recovery with restart policies, circuit breakers, and cascading failure prevention.

```python
from enum import Enum
from typing import List, Optional
import asyncio


class RestartStrategy(Enum):
    ONE_FOR_ONE = "one_for_one"  # Restart only failed agent
    ONE_FOR_ALL = "one_for_all"  # Restart all supervised agents
    REST_FOR_ONE = "rest_for_one"  # Restart failed and subsequent agents


@dataclass
class SupervisorSpec:
    """Configuration for fault tolerance supervisor."""
    max_restarts: int = 3
    restart_window_seconds: float = 60.0
    restart_strategy: RestartStrategy = RestartStrategy.ONE_FOR_ONE
    escalation_delay_seconds: float = 5.0


@dataclass 
class RestartHistory:
    """Tracks restart events for backoff calculation."""
    timestamps: List[datetime] = field(default_factory=list)
    
    def record_restart(self) -> None:
        self.timestamps.append(datetime.utcnow())
    
    def count_recent(self, window: timedelta) -> int:
        cutoff = datetime.utcnow() - window
        return sum(1 for ts in self.timestamps if ts > cutoff)


class AgentSupervisor:
    """Hierarchical supervisor managing agent lifecycle and recovery."""
    
    def __init__(self, spec: SupervisorSpec):
        self.spec = spec
        self.supervised_agents: List[Agent] = []
        self.restart_history: Dict[str, RestartHistory] = defaultdict(RestartHistory)
        self.circuit_breaker_open: Dict[str, bool] = defaultdict(bool)
        self.logger = logging.getLogger(__name__)
        
    def supervise(self, agent: Agent) -> None:
        """Add agent to supervision tree."""
        self.supervised_agents.append(agent)
        self.logger.info(f"Now supervising agent {agent.agent_id}")
    
    async def monitor_health(self) -> None:
        """Continuous health monitoring loop with failure detection."""
        while True:
            for agent in self.supervised_agents:
                if await self._is_agent_failed(agent):
                    await self._handle_failure(agent)
            
            await asyncio.sleep(1.0)
    
    async def _is_agent_failed(self, agent: Agent) -> bool:
        """Detect agent failure through state inspection."""
        if agent.state == AgentState.FAILED:
            return True
        
        # Heartbeat timeout detection
        if agent.last_heartbeat:
            timeout = timedelta(seconds=30)
            if datetime.utcnow() - agent.last_heartbeat > timeout:
                self.logger.warning(f"Agent {agent.agent_id} heartbeat timeout")
                return True
        
        return False
    
    async def _handle_failure(self, failed_agent: Agent) -> None:
        """Execute supervision strategy on agent failure."""
        agent_id = failed_agent.agent_id
        
        # Check circuit breaker
        if self.circuit_breaker_open[agent_id]:
            self.logger.error(f"Circuit breaker open for {agent_id}, "
                            "escalating failure")
            await self._escalate_failure(failed_agent)
            return
        
        # Check restart budget
        history = self.restart_history[agent_id]
        window = timedelta(seconds=self.spec.restart_window_seconds)
        recent_restarts = history.count_recent(window)
        
        if recent_restarts >= self.spec.max_restarts:
            self.logger.error(f"Max restarts exceeded for {agent_id}, "
                            "opening circuit breaker")
            self.circuit_breaker_open[agent_id] = True
            await self._escalate_failure(failed_agent)
            return
        
        # Execute restart strategy
        if self.spec.restart_strategy == RestartStrategy.ONE_FOR_ONE:
            await self._restart_agent(failed_agent)
        elif self.spec.restart_strategy == RestartStrategy.ONE_FOR_ALL:
            for agent in self.supervised_agents:
                await self._restart_agent(agent)
        elif self.spec.restart_strategy == RestartStrategy.REST_FOR_ONE:
            start_restarting = False
            for agent in self.supervised_agents:
                if agent.agent_id == agent_id:
                    start_restarting = True
                if start_restarting:
                    await self._restart_agent(agent)
        
        history.record_restart()
    
    async def _restart_agent(self, agent: Agent) -> None:
        """Restart individual agent with state recovery."""
        self.logger.info(f"Restarting agent {agent.agent_id}")
        
        # Graceful shutdown
        agent.state = AgentState.TERMINATING
        await asyncio.sleep(0.1)  # Allow cleanup
        
        # Reinitialize
        agent.state = AgentState.INITIALIZING
        agent.current_task = None
        await asyncio.sleep(self.spec.escalation_delay_seconds)
        
        agent.state = AgentState.IDLE
        agent.last_heartbeat = datetime.utcnow()
    
    async def _escalate_failure(self, agent: Agent) -> None:
        """Escalate unrecoverable failure to parent supervisor."""
        self.logger.critical(f"Escalating failure for {agent.agent_id}")
        # In production, this would notify parent supervisor or alerting system
        self.supervised_agents.remove(agent)
```

## Concept Reference

| Concept | Technical | Plain | Importance |
|---------|-----------|-------|------------|
| parallel agents | Autonomous computational entities executing concurrent processes with independent execution threads, sharing system resources while maintaining isolation | Multiple smart programs running at the same time, each doing their own work independently without waiting for others to finish | 0.95 |
| agent management | Orchestration framework providing lifecycle control, resource allocation, state monitoring, and coordination primitives for distributed autonomous entities | The system that starts, stops, monitors, and controls all the agents to make sure they work properly together | 0.92 |
| concurrency | Execution model enabling temporal overlap of computational tasks through time-slicing, multithreading, or distributed processing without strict sequencing | The ability to handle many tasks at once by switching between them quickly or running them simultaneously | 0.90 |
| distributed systems | Architectural paradigm where computational components operate across multiple networked nodes, requiring coordination protocols, consensus mechanisms | A setup where different parts of the system run on different computers or locations but work together as one system | 0.89 |
| multiple projects | Discrete organizational contexts with isolated resource boundaries, separate dependency graphs, and independent deployment pipelines requiring context switching | Different work initiatives or tasks that are kept separate from each other, each with their own goals and requirements | 0.88 |
| autonomous entities | Self-directed computational agents with decision-making capabilities, goal-oriented behavior, and reactive/proactive operational modes independent of direct control | Programs that can make their own decisions and take actions to achieve goals without someone telling them every step | 0.87 |
| inter-agent communication | Message-passing protocols enabling data exchange and coordination between distributed agents using synchronous RPC, asynchronous queues, or publish-subscribe | The ways that different agents send information and messages to each other to share updates and coordinate | 0.86 |
| resource allocation | Dynamic distribution of computational resources (CPU cycles, memory, I/O bandwidth) across competing processes using scheduling algorithms and priority management | Deciding how to fairly share computer power, memory, and other resources among all the agents that need them | 0.85 |
| scalability | System capacity to maintain performance characteristics under increasing load through horizontal scaling (adding instances) or vertical scaling (increasing capacity) | The ability to handle more and more agents or projects without slowing down or breaking | 0.84 |
| orchestration | Centralized coordination pattern managing inter-agent dependencies, workflow sequencing, and cross-cutting concerns through a supervisory control plane | A master coordinator that directs all the agents, making sure they work in the right order and coordinate their activities | 0.83 |
| fault tolerance | System resilience mechanisms including redundancy, graceful degradation, error recovery, and supervision hierarchies to maintain operational continuity under failures | The ability to keep working even when some agents fail or encounter errors, by having backups or recovery plans | 0.82 |
| state management | Persistent tracking and synchronization of agent runtime conditions, context variables, and execution history using stateful or stateless architectural patterns | Keeping track of what each agent is doing, what it knows, and where it is in its work process | 0.81 |
| load balancing | Distribution algorithm optimizing computational workload across available agents or infrastructure nodes based on capacity metrics, latency requirements | Spreading out the work evenly across all available agents so no single agent gets overwhelmed while others sit idle | 0.80 |

## Glossary

| Term | Definition | Concept IDs |
|------|------------|-------------|
| Parallel Agents | Autonomous programs executing simultaneously with independent threads, enabling concurrent task processing across system | 1, 6, 4 |
| Multi-Project Orchestration | Centralized coordination framework managing agent workflows across discrete organizational contexts with separate resources | 3, 7, 11 |
| Resource Allocation | Dynamic distribution mechanism assigning computational resources (CPU, memory, bandwidth) across competing agents using scheduling algorithms | 5, 12, 16 |
| Agent Lifecycle Control | Programmatic management of agent state transitions including initialization, activation, suspension, and termination with cleanup | 2, 14 |
| Distributed Execution | Architectural model where agents operate across networked nodes requiring coordination protocols and network-aware design | 15, 4, 10 |
| State Synchronization | Persistent tracking and coordination of agent runtime conditions, context variables, and execution history across distributed nodes | 8, 9 |
| Inter-Agent Messaging | Communication protocols enabling data exchange between distributed agents through synchronous calls, asynchronous queues, or event streams | 9, 15 |
| Context Isolation | Architectural boundaries ensuring logical separation of project execution environments to prevent cross-contamination of state or resources | 11, 3, 8 |
| Fault Resilience | System mechanisms including redundancy, error recovery, and supervision hierarchies maintaining operational continuity under component failures | 13, 14, 2 |
| Workload Distribution | Algorithms optimizing computational load across available agents based on capacity metrics, latency requirements, and fairness policies | 12, 5, 10 |
| Concurrent Execution | Computational model enabling temporal overlap of tasks through time-slicing, multithreading, or distributed processing without blocking | 4, 1, 15 |
| System Observability | Instrumentation capturing runtime metrics, distributed traces, and health indicators enabling introspection and diagnostic capabilities | 17, 2, 8 |
| Autonomous Operation | Self-directed agent capabilities including independent decision-making, goal-oriented behavior, and reactive responses without external control | 6, 1, 7 |

## Edge Cases & Warnings

- ⚠️ **Resource Starvation**: Greedy agents or projects may monopolize shared resources. Implement strict quota enforcement and fair-share scheduling policies with preemption capabilities.

- ⚠️ **Context Leakage**: Insufficient isolation boundaries may allow cross-project contamination. Enforce namespace separation, separate credential stores, and audit cross-boundary communication.

- ⚠️ **Cascading Failures**: Single agent failure may propagate through dependent agents. Implement circuit breakers, bulkhead isolation, and graceful degradation with fallback behaviors.

- ⚠️ **State Divergence**: Network partitions or message loss may cause state inconsistency. Design for eventual consistency with conflict resolution strategies and compensating transactions.

- ⚠️ **Thundering Herd**: Mass restart after system-wide failure may overwhelm infrastructure. Implement exponential backoff, jittered retry delays, and staggered restart schedules.

- ⚠️ **Deadlock Scenarios**: Circular dependencies between agents waiting on each other's completion. Use timeout-based resource acquisition and deadlock detection algorithms.

- ⚠️ **Monitoring Overhead**: Excessive observability instrumentation may degrade performance. Sample metrics strategically, aggregate before transmission, and use adaptive sampling rates.

- ⚠️ **Scale-Down Safety**: Terminating agents with in-flight work causes data loss. Drain connections gracefully, persist checkpoints, and wait for work completion before shutdown.

## Quick Reference

```python
# Minimal end-to-end orchestration example
import asyncio
from datetime import timedelta

async def main():
    # Initialize orchestrator with total system resources
    orchestrator = MultiProjectOrchestrator(
        total_resources={"cpu": 8.0, "memory_gb": 32.0}
    )
    
    # Register multiple isolated project contexts
    project_a_context = ProjectContext(
        project_id="analytics_pipeline",
        resource_quota={"cpu": 4.0, "memory_gb": 16.0},
        max_agents=5,
        isolation_namespace="proj_a"
    )
    pool_a = orchestrator.register_project(project_a_context)
    
    project_b_context = ProjectContext(
        project_id="ml_training",
        resource_quota={"cpu": 4.0, "memory_gb": 16.0},
        max_agents=3,
        isolation_namespace="proj_b"
    )
    pool_b = orchestrator.register_project(project_b_context)
    
    # Submit workload to each project
    for i in range(10):
        await orchestrator.submit_task_to_project(
            "analytics_pipeline", f"analyze_batch_{i}"
        )
        await orchestrator.submit_task_to_project(
            "ml_training", f"train_model_{i}"
        )
    
    # Start task processing with automatic load balancing
    processing_tasks = [
        asyncio.create_task(pool_a.process_tasks()),
        asyncio.create_task(pool_b.process_tasks())
    ]
    
    # Monitor system status
    await asyncio.sleep(2)
    status = orchestrator.get_system_status()
    for allocation in status:
        print(f"{allocation.project_id}: "
              f"{allocation.active_agents} agents, "
              f"{allocation.pending_tasks} pending")
    
    # Cleanup
    for task in processing_tasks:
        task.cancel()

# Run orchestration
asyncio.run(main())
```

---
_Generated by Philosopher's Stone v4 — EchoSeed_
Philosopher's Stone v4 × Skill Forge × EchoSeed
