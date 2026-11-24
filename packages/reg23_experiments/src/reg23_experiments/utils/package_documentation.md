Helpers that know nothing about registration.

| Module            | Purpose                                                                                    |
|-------------------|--------------------------------------------------------------------------------------------|
| `logs_setup`      | Sets up logging for a script                                                               |
| `console_logging` | Coloured, width-aware log formatting that works alongside progress bars                    |
| `pushover`        | Sends a notification to a phone                                                            |
| `reflection`      | Inspects function arguments                                                                |
| `data`            | Copies and observes nested `traitlets` objects                                             |

## Logging

Scripts call `logs_setup.setup_logger()` once. It reads `logging.conf` from the working directory, so scripts should be run
from the repository root, and writes to the terminal and to a file in `Logs/` whose name includes the host, time and script.
Library modules use `logging.getLogger(__name__)`.

Use `tqdm` from `console_logging` for progress bars. It is disabled when the output is not a terminal, and log messages
appear above the bar rather than through it. `indentation_prefix` makes nested bars readable.

## Notifications

`pushover.send_notification` reads `PUSHOVER_USER_ID` and `PUSHOVER_API_TOKEN` from the environment, and logs an error if they
are missing.

## Reflection

`FunctionArgument.get_for_function` lists a function's arguments and whether they are required. The data manager uses it to
work out which graph nodes a function depends on.
