set dotenv-load := true

build platform="cuda":
    uv sync --extra {{platform}} --no-install-workspace
    uv sync --extra {{platform}}

devbuild platform="cuda":
    uv sync --extra {{platform}} --extra dev --no-install-workspace
    uv sync --extra {{platform}} --extra dev

optional_name(name) := if name == "" { "" } else { "--name " + name }

experiment ctpath xraydir name="" platform="cuda":
    uv run --extra {{platform}} --extra dev scripts/program_truncation.py \
        --ct-path {{ctpath}} \
        --xray-dir {{xraydir}} \
        --notify \
        {{optional_name(name)}}

runapp platform="cuda":
    uv run --extra {{platform}} scripts/app.py

devapp platform="cuda":
    uv run --extra {{platform}} --extra dev scripts/app.py
