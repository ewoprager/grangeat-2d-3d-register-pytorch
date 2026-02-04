- `state.py` contains the implementation for `AppState`, which stores all the internal state of the app, and manages the
  relationship between the global data manager (DAG) and the interface parameters.
- `controller.py` contains the implementation for `Controller`, which manages workers in separate threads. It only
  communicates with workers via signals. It can read and mutate the app state.
- `workers/` contains implementations of workers, which do not interact with the rest of the app's implementation
  directly, only with the controller via signals.
- `gui/` contains implementations of GUI elements, which only read and write app state. Writing of the app state by the
GUI is done primarily through update signals from the GUI elements to the parameters in the state.