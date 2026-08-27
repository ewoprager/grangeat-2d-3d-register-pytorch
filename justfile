set dotenv-load := true

build platform="cuda":
    uv sync --extra {{platform}} --no-build-isolation --no-install-workspace
    uv sync --extra {{platform}} --no-build-isolation
    
devbuild platform="cuda":
    uv sync --extra {{platform}} --extra dev --no-build-isolation --no-install-workspace
    uv sync --extra {{platform}} --extra dev --no-build-isolation

experiment ctpath xraydir name="" platform="cuda":
    uv run --extra {{platform}} --extra dev --no-build-isolation scripts/program_truncation.py \
        --ct-path "{{ctpath}}" \
        --xray-dir "{{xraydir}}" \
        --notify \
        {{ if name == "" { "" } else { "--name " + name } }}

runapp platform="cuda":
    uv run --extra {{platform}} --no-build-isolation scripts/app.py

devapp platform="cuda":
    uv run --extra {{platform}} --extra dev --no-build-isolation scripts/app.py
