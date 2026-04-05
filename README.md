# Quoridor 4P

A Python implementation of Quoridor focused on course-delivery priorities: correct core rules first, local play and AI second, online MVP third.

## Included
- Shared core engine for local, AI, save/load, and online.
- Local pygame client with main menu, new game presets, Continue, Save, pause overlay, and non-blocking local AI turns and async wall preview calculation.
- AI implementations: Random, Minimax for 2-player mode, and MCTS for multiplayer mode.
- Online MVP: authoritative websocket server with room join, ready, state sync, and game-over broadcast.
- Unit tests for core rules, AI legality, and room lifecycle.

## Project Layout
- `quoridor/core`: state, actions, rules, pathfinding, serialization.
- `quoridor/ai`: random, minimax, MCTS, action pruning, heuristics.
- `quoridor/client`: pygame application and online websocket session.
- `quoridor/server`: room manager, protocol helpers, websocket server.
- `scripts`: runnable entrypoints.
- `tests`: unit tests.

## Requirements
Install dependencies in a local environment:

```bash
python -m pip install -r requirements.txt
```

Required runtime packages:
- `pygame`
- `websockets`

## Run Local Client
```bash
python scripts/run_client.py
```

Menu presets include:
- `Local 2P`
- `Local 4P`
- `Vs Random`
- `Vs Minimax`
- `Vs MCTS`
- `Continue`
- `Online`

## Controls
- Left click highlighted cells to move.
- Use `Move`, `Wall H`, and `Wall V` buttons to switch action type.
- Press `Esc` to open the pause overlay.
- `Save` writes the current local match to `saves/latest_local_game.json`.
- `Continue` loads the most recent local save.

## Run Online MVP
Start the server:

```bash
python scripts/run_server.py
```

Then open the client on up to 4 machines or windows:

```bash
python scripts/run_client.py
```

On the online screen:
- Enter a server URL, name, and optional room code.
- Leave room code blank to create a new room.
- Share the generated room code with the other players.
- After 4 players join, each player presses `Ready`.

## Run Self-Play
```bash
python scripts/selfplay.py --turn-limit 120
```

## Run Tests
```bash
python -m unittest discover -s tests -v
```

## Notes
- Minimax is intentionally limited to 2-player games.
- Save/Continue is intentionally local-only.
- The online mode is an MVP: it has no reconnect, persistence, login, or anti-cheat beyond authoritative validation.


