- `state.py` contains the implementation for `AppState`, which stores all the internal state of the app
- `context.py` contains the implementation for `AppContext`, which is received by most app functionality. It owns
    - the instance of `AppState`,
    - the instance of the parent `DADG`, and
    - services like save data managers.
- `worker_manager.py` contains the implementation for `WorkerManager`, which manages workers in separate threads. It
  only communicates with workers via signals. It can read and mutate the app state.
- `workers/` contains implementations of workers, which do not interact with the rest of the app's implementation
  directly, only with the controller via signals.
- `gui/` contains implementations of GUI elements, which only read and write app state. Writing of the app state by the
  GUI is done primarily through update signals from the GUI elements to the parameters in the state.