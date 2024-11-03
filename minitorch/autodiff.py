from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """

    vals = list(vals)
    vals[arg] += epsilon / 2
    f_r = f(*vals)
    vals[arg] -= epsilon
    f_l = f(*vals)
    return (f_r - f_l) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # graph (variable parents etc.) should be acyclic
    out = []
    vis = set()

    def dfs(var: Variable):
        if var.is_constant():
            return
        vis.add(var.unique_id)
        for parent in var.parents:
            if parent.unique_id not in vis:
                dfs(parent)
        out.append(var)
        return

    dfs(variable)
    return out[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    1. Call topological sort to get an ordered queue
    2. Create a dictionary of Scalars and current derivatives
    3. For each node in backward order, pull a completed Scalar and derivative from the queue:
        a. if the Scalar is a leaf, add its final derivative (`accumulate_derivative`) and loop to (1)
        b. if the Scalar is not a leaf,
            i. call `.chain_rule` on the last function with $d_out$
            ii. loop through all the Scalars+derivative produced by the chain rule
            iii. accumulate derivatives for the Scalar in a dictionary
    """

    q: Iterable[Variable] = topological_sort(variable)

    derivatives = {variable.unique_id: deriv}
    for scalar in q:
        d_out = derivatives[scalar.unique_id]
        if scalar.is_leaf():
            scalar.accumulate_derivative(d_out)
        else:
            for var, derivative in scalar.chain_rule(d_out):
                if var.unique_id in derivatives:
                    derivatives[var.unique_id] += derivative
                else:
                    derivatives[var.unique_id] = derivative
    return


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
