# Gemini CLI Project Rules: CryptoTrader

These instructions are foundational mandates for Gemini CLI and take absolute precedence over general defaults.

## Python Standards
- **Typing:** Mandatory PEP 484 type hints for all function signatures and class members.
- **Documentation:** Use Google-style docstrings for all classes and public methods.
- **Linting:** Adhere to PEP 8 standards. Prefer `ruff` for linting and formatting if available.
- **Testing:** Use `pytest` for all unit and integration tests. Every new feature or bug fix must include a corresponding test.

## Domain Specifics (Reinforcement Learning & Trading)
- **Math & Performance:** Always use `numpy` for vectorized operations and state calculations. Avoid raw Python loops for data processing.
- **Gymnasium API:** Strictly follow the latest `gymnasium` interface:
  - `reset()` must return `(observation, info)`.
  - `step()` must return `(observation, reward, terminated, truncated, info)`.
- **Financial Precision:** Use high-precision `float` or `decimal.Decimal` for balance calculations to minimize rounding errors during backtesting.
- **Action Safety:** Implement robust checks or action masking in environments to prevent illegal operations (e.g., buying with insufficient funds).

## Security & Secrets
- **Credential Protection:** NEVER hardcode API keys, secrets, or sensitive configuration. Use environment variables or `.env` files.
- **Git Safety:** Ensure `.env`, `logs/`, and large data files are excluded via `.gitignore`.

## Agent Skills (Superpowers)
- **Using Superpowers:** Establish how to find and use skills by invoking Skill tools before responding.
- **Brainstorming:** MUST be used before any creative work to explore intent, requirements, and design.
- **Writing Plans:** Create a detailed spec/requirement plan for multi-step tasks before touching code.
- **Test-Driven Development:** Implement tests before writing implementation code for any feature or bugfix.
- **Systematic Debugging:** Use when encountering bugs or test failures to investigate before proposing fixes.
- **Verification Before Completion:** Evidence before assertions. Run verification commands and confirm output before claiming work is complete.
- **Executing Plans:** Use written implementation plans with review checkpoints in separate sessions.
- **Subagent & Parallel Development:** Use subagents or parallel agents for independent tasks without shared state.
- **Git Worktrees:** Use for feature isolation and implementation safety.
- **Code Review (Requesting & Receiving):** Use to verify requirements and maintain technical rigor; verify feedback before implementation.
- **Finishing Branches:** Use structured options (merge, PR, cleanup) when implementation and tests are complete.
- **Skill Management:** Use `writing-skills` for creating, editing, or verifying agent skills.
