MTR-OR-Agent: Explainable Model Reasoning for Metro Scheduling

 This project is the implementation code for a Master's Dissertation exploring the integration of Large Language Models (LLMs) with deterministic Operations Research (OR) solvers for urban rail transit scheduling.


## Overview

Applying zero-shot LLMs directly to physical operations often triggers severe "variable hallucinations." The **OR-LLM-Agent** solves this by decoupling the optimization lifecycle into function-specific agents (`MathAgent`, `CodeAgent`, `DebuggingAgent`). Through an isolated sandbox execution environment, the framework translates natural language queries into rigorous `Gurobi` solver states.

## Core Architecture

- **Multi-Model Routing:** Built-in lightweight HTTP clients supporting Qwen-max, DeepSeek-chat, and GPT-5.2.
- **Autonomous Sandbox Debugging:** Executes generated Gurobi scripts in an isolated process with an automated, up-to-5-iteration self-repair mechanism.
- **Physical Boundary Verification:** Rejects mathematically optimal but physically impossible schedules (e.g., violating signaling system limits).

## Quick Start

### 1. Installation
Clone the repository and install the dependencies
cd OR-LLM-Agent
pip install -r requirements.txt

### 2. Prerequisites
- A valid [Gurobi Optimizer License](https://www.gurobi.com/) must be installed on your machine.
- You will be prompted to enter your LLM API Key (Qwen / DeepSeek / OpenAI-compatible) when running the system.

### 3. Run the Interactive CLI
The system provides a terminal-based interactive interface for testing scheduling scenarios:
```bash
python model_v21.py
```
*Example Prompt:* "荃湾线客流8500人/小时，列车容量1500人，求最优发车间隔"

## Known Limitations & Future Work
As an academic prototype, the current implementation has the following boundaries:
1. **Static Topologies:** The HK Metro Data loader currently utilizes static pre-defined dictionaries for distances and capacities.
2. **Rule-based Linearization:** The non-linear to linear constraint conversion (e.g., transforming capacity division into multiplication) relies on regex-based placeholder rules rather than full Abstract Syntax Tree (AST) parsing.
3. **Execution Latency:** Sandbox creation and repetitive API calls introduce latency not suitable for millisecond-level real-time control.


> **⚠️ Academic Disclaimer (Proof of Concept):** 
> Please note that this repository serves as an academic prototype (Proof of Concept). It is designed to validate the theoretical multi-agent workflow proposed in the dissertation. Certain modules (such as comprehensive real-time network dynamic routing) contain simplified heuristics or placeholder logic to facilitate the core experiments. It still have gap for production-level industrial deployment.

