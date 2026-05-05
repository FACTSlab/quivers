"""Tests for traced monoidal categories."""
import torch
import pytest
from quivers.core.objects import FinSet, ProductSet
from quivers.core.morphisms import observed, identity
from quivers.core.quantales import BOOLEAN
from quivers.categorical.monoidal import CartesianMonoidal
from quivers.categorical.traced import CartesianTrace, IterativeTrace, trace, partial_trace

class TestCartesianTrace:

    def test_trace_of_identity(self):
        """Tr^U(id_{A×U}) should give a morphism with all entries joined."""
        a = FinSet(name='A', cardinality=2)
        u = FinSet(name='U', cardinality=2)
        au = ProductSet(components=(a, u))
        id_au = identity(au)
        tracer = CartesianTrace()
        result = tracer.trace(id_au, u, a, a)
        assert result.tensor.shape == (2, 2)
        expected = identity(a).tensor
        torch.testing.assert_close(result.tensor, expected, atol=0.0001, rtol=0.0001)

    def test_trace_of_swap(self):
        """Trace of swap σ_{U,U}: U×U → U×U should be id_U (yanking)."""
        u = FinSet(name='U', cardinality=3)
        ProductSet(components=(u, u))
        monoidal = CartesianMonoidal()
        swap = monoidal.braiding(u, u)
        tracer = CartesianTrace()
        result = tracer.trace(swap, u, u, u)
        expected = identity(u).tensor
        torch.testing.assert_close(result.tensor, expected, atol=0.0001, rtol=0.0001)

    def test_yanking_axiom(self):
        """Verify yanking: Tr(σ) = id."""
        tracer = CartesianTrace()
        u = FinSet(name='U', cardinality=3)
        assert tracer.verify_yanking(u)

    def test_trace_convenience_function(self):
        """The trace() function should match CartesianTrace."""
        a = FinSet(name='A', cardinality=2)
        u = FinSet(name='U', cardinality=2)
        au = ProductSet(components=(a, u))
        id_au = identity(au)
        result = trace(id_au, u, a, a)
        assert result.tensor.shape == (2, 2)

class TestIterativeTrace:

    def test_matches_cartesian(self):
        """Iterative trace should match cartesian for simple cases."""
        a = FinSet(name='A', cardinality=2)
        u = FinSet(name='U', cardinality=2)
        au = ProductSet(components=(a, u))
        id_au = identity(au)
        ct = CartesianTrace()
        it = IterativeTrace(CartesianMonoidal())
        result_ct = ct.trace(id_au, u, a, a).tensor
        result_it = it.trace(id_au, u, a, a).tensor
        torch.testing.assert_close(result_ct, result_it, atol=0.0001, rtol=0.0001)

class TestPartialTrace:

    def test_partial_trace_single_component(self):
        """Partial trace over one component of a product."""
        a = FinSet(name='A', cardinality=2)
        b = FinSet(name='B', cardinality=2)
        u = FinSet(name='U', cardinality=2)
        dom = ProductSet(components=(a, u))
        cod = ProductSet(components=(b, u))
        ProductSet(components=(a, u, b, u))
        data = torch.zeros(*dom.shape, *cod.shape)
        for ai in range(2):
            for ui in range(2):
                for bi in range(2):
                    for uo in range(2):
                        if ai == bi and ui == uo:
                            data[ai, ui, bi, uo] = 1.0
        morph = observed(dom, cod, data)
        result = partial_trace(morph, feedback_indices=(1,))
        assert result.tensor.shape == (2, 2)

    def test_partial_trace_requires_product(self):
        """Should raise for non-ProductSet domain."""
        a = FinSet(name='A', cardinality=2)
        f = identity(a)
        with pytest.raises(TypeError, match='ProductSet'):
            partial_trace(f, feedback_indices=(0,))

class TestBooleanTrace:

    def test_boolean_trace(self):
        """Trace should work with boolean quantale."""
        a = FinSet(name='A', cardinality=2)
        u = FinSet(name='U', cardinality=2)
        au = ProductSet(components=(a, u))
        id_au = identity(au, quantale=BOOLEAN)
        tracer = CartesianTrace(quantale=BOOLEAN)
        result = tracer.trace(id_au, u, a, a)
        expected = identity(a, quantale=BOOLEAN).tensor
        torch.testing.assert_close(result.tensor, expected, atol=1e-05, rtol=1e-05)