set dotenv-load

build platform="cuda":
    uv sync --extra {{ platform }} --no-install-workspace
    uv sync --extra {{ platform }}

devbuild platform="cuda":
    uv sync --extra {{ platform }} --extra dev --no-install-workspace
    uv sync --extra {{ platform }} --extra dev

experiment ctpath xraydir name="" platform="cuda":
    uv run --extra {{ platform }} --extra dev scripts/program_truncation.py \
        --ct-path "{{ ctpath }}" \
        --xray-dir "{{ xraydir }}" \
        --notify \
        {{ if name == "" { "" } else { "--name \"" + name + "\"" } }}

runapp platform="cuda":
    uv run --extra {{ platform }} -m reg23_app

devapp platform="cuda":
    uv run --extra {{ platform }} --extra dev -m reg23_app

gen_starting_positions ctpath xraypath count=50:
    uv run --extra dev -m reg23_experiments.experiments.reg_experiment.generate_starting_positions \
        --ct-path "{{ ctpath }}"
