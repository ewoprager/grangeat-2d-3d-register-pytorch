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


def test_child_dag():
    dag = DAG()
    dag.add_updater("a_plus_b_equals_c", a_plus_b_equals_c)
    dag.set_data_multiple(a=1.0, b=2.0)
    child_dag = ChildDAG(dag)
    assert dag.get("c") == pytest.approx(3.0)
    assert child_dag.get("c") == pytest.approx(3.0)
    print(f"dag:\n{dag}\nchild:\n{child_dag}")

    child_dag.set_data("b", 2.1)
    assert dag.get("c") == pytest.approx(3.0)
    assert child_dag.get("c") == pytest.approx(3.1)
    print(f"dag:\n{dag}\nchild:\n{child_dag}")

    dag.set_data("a", 1.2)
    assert dag.get("c") == pytest.approx(3.2)
    assert child_dag.get("c") == pytest.approx(3.1)
    print(f"dag:\n{dag}\nchild:\n{child_dag}")
