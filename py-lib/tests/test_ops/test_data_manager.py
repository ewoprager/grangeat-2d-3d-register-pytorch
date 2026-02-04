from typing import Any

import pytest

from reg23_experiments.ops.data_manager import DAG, ChildDAG, dag_updater


@dag_updater(names_returned=["c"])
def a_plus_b_equals_c(a: float, b: float) -> dict[str, Any]:
    return {"c": a + b}


def test_dag():
    dag = DAG()
    dag.add_updater("a_plus_b_equals_c", a_plus_b_equals_c)
    dag.set_data_multiple(a=1.0, b=2.0)
    assert dag.get("c") == pytest.approx(3.0)
