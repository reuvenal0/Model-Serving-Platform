# AI Usage

## 1. Tools I Used

I used **ChatGPT** and **OpenAI Codex** throughout the development of the Model Serving Platform.

- **ChatGPT:** Helped me discuss architectural alternatives, challenge design decisions, and refine the overall development plan.
- **OpenAI Codex:** Assisted with implementing the code, writing automated tests, debugging, documentation, and setting up Docker and GitHub Actions CI.

I used AI as a development assistant, while retaining responsibility for the architecture, technical decisions, and validation of the results.

## 2. What Helped Most

The biggest benefit was **accelerating implementation**. Codex helped translate architectural decisions into working code and tests, allowing me to build a project of this scope in approximately 48 hours.

I also found architectural discussions with ChatGPT valuable. I started with my own design ideas, compared them against AI suggestions, challenged recommendations when necessary, and selected the approaches I considered most appropriate.

My main design priority was **modularity and extensibility**. I wanted a working platform built around abstractions, interfaces, dependency injection, and interchangeable implementations, rather than a solution tightly coupled to specific frameworks or infrastructure.

## 3. What I Had to Fix

One example was the initial database migration approach. AI-assisted implementation introduced support for historical schema migrations, but I questioned whether that complexity was justified for a project still under internal development, with no existing customers or production databases to support.

I chose to simplify fresh database initialization rather than maintain unnecessary compatibility with intermediate development schemas.

Another example was the demonstration experience. I pushed for a self-contained demo with sample models and predictable outputs, so reviewers could run and evaluate the platform without preparing their own training data or models.

These decisions reflected my responsibility to keep the implementation aligned with the project's actual goals.

## 4. What AI Struggled With

AI sometimes had difficulty distinguishing between the local development environment and Docker execution, particularly regarding configuration, networking, and access restrictions.

It also occasionally suggested solutions that were unnecessarily complex or restrictive for the project's scope. For example, I wanted a convenient development workflow using FastAPI's Swagger UI at `/docs`, while preserving appropriate security boundaries for operational endpoints.

This required explicitly introducing a development-mode configuration instead of accepting a setup that made manual API exploration unnecessarily difficult.

Overall, the main challenge was **balancing technical correctness, security, simplicity, and developer experience**. AI was effective at generating implementations, but deciding which complexity was justified remained my responsibility.