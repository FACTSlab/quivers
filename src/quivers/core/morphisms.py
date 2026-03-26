"""Morphism hierarchy: V-enriched relations between finite sets.

A morphism from domain D to codomain C is represented as a tensor of
shape (*D.shape, *C.shape) with entries in the lattice L of a quantale.
Composition uses the quantale's operations (join over tensor product).

The hierarchy:

    Morphism (abstract)
    ├── ObservedMorphism    — fixed tensor (not learned)
    ├── LatentMorphism      — nn.Parameter with sigmoid constraint
    ├── ComposedMorphism    — f >> g (V-enriched composition)
    ├── ProductMorphism     — f @ g (tensor / parallel product)
    ├── MarginalizedMorphism — contract codomain dims via join
    ├── FunctorMorphism     — lazy image of a morphism under a functor
    └── RepeatMorphism      — runtime-variable iterated composition (T^n)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from quivers.core.objects import SetObject, ProductSet
from quivers.core.quantales import PRODUCT_FUZZY, Quantale

if TYPE_CHECKING:
    from quivers.categorical.functors import Functor


class Morphism(ABC):
    """Abstract base for morphisms between finite sets.

    Subclasses must implement ``tensor`` (returns the materialized
    tensor with values in the quantale's lattice) and ``module``
    (returns the nn.Module tree for parameter collection).

    Parameters
    ----------
    domain : SetObject
        Source object.
    codomain : SetObject
        Target object.
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.
    """

    def __init__(
        self,
        domain: SetObject,
        codomain: SetObject,
        quantale: Quantale | None = None,
    ) -> None:
        self._domain = domain
        self._codomain = codomain
        self._quantale = quantale if quantale is not None else PRODUCT_FUZZY

    @property
    def domain(self) -> SetObject:
        """Source object."""
        return self._domain

    @property
    def codomain(self) -> SetObject:
        """Target object."""
        return self._codomain

    @property
    def quantale(self) -> Quantale:
        """The enrichment algebra for this morphism."""
        return self._quantale

    @property
    def tensor_shape(self) -> tuple[int, ...]:
        """Expected shape of the materialized tensor."""
        return (*self._domain.shape, *self._codomain.shape)

    @property
    @abstractmethod
    def tensor(self) -> torch.Tensor:
        """Materialize the morphism as a tensor with values in L."""
        ...

    @abstractmethod
    def module(self) -> nn.Module:
        """Return an nn.Module wrapping all learnable parameters."""
        ...

    # -- dsl operators -------------------------------------------------------

    def __rshift__(self, other: Morphism) -> ComposedMorphism:
        """V-enriched composition: self >> other.

        Composes self: A -> B with other: B -> C to yield
        a morphism A -> C, contracting over B using the quantale.

        Parameters
        ----------
        other : Morphism
            Right morphism whose domain must match self's codomain.

        Returns
        -------
        ComposedMorphism
            The composed morphism.
        """
        if not isinstance(other, Morphism):
            return NotImplemented

        if self.codomain != other.domain:
            raise TypeError(
                f"cannot compose: codomain {self.codomain!r} != domain {other.domain!r}"
            )

        if not self._quantale.is_compatible(other._quantale):
            raise TypeError(
                f"incompatible quantales: {self._quantale!r} and {other._quantale!r}"
            )

        return ComposedMorphism(self, other)

    def __matmul__(self, other: Morphism) -> ProductMorphism:
        """Tensor (parallel) product: self @ other.

        Given self: A -> B and other: C -> D, produces
        a morphism A×C -> B×D whose tensor is the outer product
        via the quantale's tensor_op.

        Parameters
        ----------
        other : Morphism
            Right morphism.

        Returns
        -------
        ProductMorphism
            The product morphism.
        """
        if not isinstance(other, Morphism):
            return NotImplemented

        return ProductMorphism(self, other)

    def marginalize(self, *sets: SetObject) -> MarginalizedMorphism:
        """Marginalize (join-reduce) over codomain components.

        The codomain must be a ProductSet containing the sets to
        marginalize over. The result has a codomain with those
        components removed. Uses the quantale's join operation.

        Parameters
        ----------
        *sets : SetObject
            Codomain components to marginalize over.

        Returns
        -------
        MarginalizedMorphism
            The marginalized morphism.
        """
        return MarginalizedMorphism(self, sets)

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}({self.domain!r} -> {self.codomain!r})"


# -- leaf morphisms ----------------------------------------------------------


class _MorphismModule(nn.Module):
    """Internal nn.Module wrapper for a single morphism's parameters."""

    pass


class ObservedMorphism(Morphism):
    """A morphism with a fixed (non-learnable) tensor.

    Parameters
    ----------
    domain : SetObject
        Source object.
    codomain : SetObject
        Target object.
    data : torch.Tensor
        Fixed tensor of shape (*domain.shape, *codomain.shape).
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.
    """

    def __init__(
        self,
        domain: SetObject,
        codomain: SetObject,
        data: torch.Tensor,
        quantale: Quantale | None = None,
    ) -> None:
        super().__init__(domain, codomain, quantale=quantale)

        expected = self.tensor_shape

        if data.shape != expected:
            raise ValueError(
                f"data shape {data.shape} does not match expected {expected}"
            )

        self._module = _MorphismModule()
        self._module.register_buffer("data", data)

    @property
    def tensor(self) -> torch.Tensor:
        return self._module.data

    def module(self) -> nn.Module:
        return self._module


class LatentMorphism(Morphism):
    """A learnable morphism backed by an nn.Parameter.

    Stores unconstrained real-valued parameters and applies sigmoid
    to produce [0,1]-valued fuzzy relation entries.

    Parameters
    ----------
    domain : SetObject
        Source object.
    codomain : SetObject
        Target object.
    init_scale : float
        Standard deviation of the normal initialization for the
        unconstrained parameters. Default 0.5 (sigmoid maps this
        to roughly uniform over [0.3, 0.7]).
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.
    """

    def __init__(
        self,
        domain: SetObject,
        codomain: SetObject,
        init_scale: float = 0.5,
        quantale: Quantale | None = None,
    ) -> None:
        super().__init__(domain, codomain, quantale=quantale)

        shape = self.tensor_shape
        self._module = _MorphismModule()
        raw = nn.Parameter(torch.randn(shape) * init_scale)
        self._module.register_parameter("raw", raw)

    @property
    def raw(self) -> nn.Parameter:
        """Unconstrained parameter tensor."""
        return self._module.raw  # type: ignore[return-value]

    @property
    def tensor(self) -> torch.Tensor:
        """Sigmoid-constrained tensor with values in (0, 1)."""
        return torch.sigmoid(self._module.raw)

    def module(self) -> nn.Module:
        return self._module


# -- composite morphisms -----------------------------------------------------


class _ComposedModule(nn.Module):
    """Module for composed morphisms that owns sub-modules."""

    def __init__(self, left_mod: nn.Module, right_mod: nn.Module) -> None:
        super().__init__()
        self.left = left_mod
        self.right = right_mod


class ComposedMorphism(Morphism):
    """V-enriched composition of two morphisms.

    Given left: A -> B and right: B -> C, produces A -> C
    by contracting over B using the quantale's compose method.

    Parameters
    ----------
    left : Morphism
        Left morphism (applied first).
    right : Morphism
        Right morphism (applied second).
    """

    def __init__(self, left: Morphism, right: Morphism) -> None:
        # the shared set is left.codomain = right.domain
        n_contract = left.codomain.ndim
        super().__init__(left.domain, right.codomain, quantale=left._quantale)
        self._left = left
        self._right = right
        self._n_contract = n_contract

    @property
    def left(self) -> Morphism:
        """Left (first) morphism."""
        return self._left

    @property
    def right(self) -> Morphism:
        """Right (second) morphism."""
        return self._right

    @property
    def tensor(self) -> torch.Tensor:
        return self._quantale.compose(
            self._left.tensor,
            self._right.tensor,
            self._n_contract,
        )

    def module(self) -> nn.Module:
        return _ComposedModule(
            self._left.module(),
            self._right.module(),
        )


class _ProductModule(nn.Module):
    """Module for product morphisms."""

    def __init__(self, left_mod: nn.Module, right_mod: nn.Module) -> None:
        super().__init__()
        self.left = left_mod
        self.right = right_mod


class ProductMorphism(Morphism):
    """Tensor (parallel) product of two morphisms.

    Given left: A -> B and right: C -> D, produces
    A×C -> B×D. The tensor is the outer product of the
    two component tensors via the quantale's tensor_op.

    Parameters
    ----------
    left : Morphism
        Left morphism.
    right : Morphism
        Right morphism.
    """

    def __init__(self, left: Morphism, right: Morphism) -> None:
        domain = ProductSet(left.domain, right.domain)
        codomain = ProductSet(left.codomain, right.codomain)
        super().__init__(domain, codomain, quantale=left._quantale)
        self._left = left
        self._right = right

    @property
    def tensor(self) -> torch.Tensor:
        lt = self._left.tensor
        rt = self._right.tensor

        # lt has shape (*dom_l, *cod_l)
        # rt has shape (*dom_r, *cod_r)
        # result should have shape (*dom_l, *dom_r, *cod_l, *cod_r)
        #
        # we need to interleave the dimensions properly.
        # step 1: compute outer product via quantale tensor_op
        n_l = lt.ndim
        n_r = rt.ndim

        # expand lt: (*dom_l, *cod_l, *[1]*n_r)
        lt_expanded = lt.reshape(*lt.shape, *([1] * n_r))

        # expand rt: (*[1]*n_l, *dom_r, *cod_r)
        rt_expanded = rt.reshape(*([1] * n_l), *rt.shape)

        # outer product via quantale: (*dom_l, *cod_l, *dom_r, *cod_r)
        outer = self._quantale.tensor_op(lt_expanded, rt_expanded)

        # now permute to (*dom_l, *dom_r, *cod_l, *cod_r)
        n_dom_l = self._left.domain.ndim
        n_cod_l = self._left.codomain.ndim
        n_dom_r = self._right.domain.ndim

        # current layout: [dom_l dims] [cod_l dims] [dom_r dims] [cod_r dims]
        # target layout:  [dom_l dims] [dom_r dims] [cod_l dims] [cod_r dims]
        dom_l_dims = list(range(n_dom_l))
        cod_l_dims = list(range(n_dom_l, n_dom_l + n_cod_l))
        dom_r_dims = list(range(n_dom_l + n_cod_l, n_dom_l + n_cod_l + n_dom_r))
        cod_r_dims = list(range(n_dom_l + n_cod_l + n_dom_r, n_l + n_r))

        perm = dom_l_dims + dom_r_dims + cod_l_dims + cod_r_dims

        return outer.permute(*perm)

    def module(self) -> nn.Module:
        return _ProductModule(
            self._left.module(),
            self._right.module(),
        )


class _MarginalizedModule(nn.Module):
    """Module for marginalized morphisms."""

    def __init__(self, inner_mod: nn.Module) -> None:
        super().__init__()
        self.inner = inner_mod


class MarginalizedMorphism(Morphism):
    """Morphism with codomain dimensions marginalized via the quantale's join.

    Given an inner morphism A -> B × C and a set B to marginalize,
    produces A -> C by join-reduction over B's dimensions.

    Parameters
    ----------
    inner : Morphism
        The morphism to marginalize.
    sets_to_marginalize : tuple of SetObject
        Codomain components to marginalize over.
    """

    def __init__(
        self,
        inner: Morphism,
        sets_to_marginalize: tuple[SetObject, ...] | list[SetObject],
    ) -> None:
        codomain = inner.codomain
        sets_to_marginalize = tuple(sets_to_marginalize)

        # find which codomain dimensions to contract
        if not isinstance(codomain, ProductSet):
            raise TypeError(
                f"can only marginalize over ProductSet codomain, "
                f"got {type(codomain).__name__}"
            )

        # identify dimensions to marginalize
        # the tensor has shape (*domain.shape, *codomain.shape)
        # codomain.shape comes from the product's components
        n_domain = inner.domain.ndim
        remaining_components: list[SetObject] = []
        dims_to_reduce: list[int] = []
        offset = n_domain

        for component in codomain.components:
            if component in sets_to_marginalize:
                # mark these dimensions for reduction
                for d in range(component.ndim):
                    dims_to_reduce.append(offset + d)

            else:
                remaining_components.append(component)

            offset += component.ndim

        if not dims_to_reduce:
            raise ValueError("none of the specified sets found in codomain components")

        # build new codomain
        if len(remaining_components) == 0:
            raise ValueError("cannot marginalize all codomain components")

        elif len(remaining_components) == 1:
            new_codomain = remaining_components[0]

        else:
            new_codomain = ProductSet(*remaining_components)

        super().__init__(inner.domain, new_codomain, quantale=inner._quantale)
        self._inner = inner
        self._dims_to_reduce = tuple(dims_to_reduce)

    @property
    def tensor(self) -> torch.Tensor:
        return self._quantale.join(
            self._inner.tensor,
            dim=self._dims_to_reduce,
        )

    def module(self) -> nn.Module:
        return _MarginalizedModule(self._inner.module())


class FunctorMorphism(Morphism):
    """Lazy image of a morphism under a functor.

    Recomputes the tensor on each access from the inner morphism,
    preserving gradient flow through the functor's map_tensor method.
    No additional parameters beyond those of the inner morphism.

    Parameters
    ----------
    functor : Functor
        The functor that produced this morphism.
    inner : Morphism
        The original morphism being mapped.
    domain : SetObject
        The image of the inner morphism's domain under the functor.
    codomain : SetObject
        The image of the inner morphism's codomain under the functor.
    """

    def __init__(
        self,
        functor: Functor,
        inner: Morphism,
        domain: SetObject,
        codomain: SetObject,
    ) -> None:
        super().__init__(domain, codomain, quantale=inner._quantale)
        self._functor = functor
        self._inner = inner

    @property
    def inner(self) -> Morphism:
        """The original morphism being mapped."""
        return self._inner

    @property
    def tensor(self) -> torch.Tensor:
        return self._functor.map_tensor(self._inner.tensor, self._quantale)

    def module(self) -> nn.Module:
        # same parameters as the inner morphism
        return self._inner.module()


class _RepeatModule(nn.Module):
    """Module wrapper for RepeatMorphism parameters."""

    def __init__(self, inner_mod: nn.Module) -> None:
        super().__init__()
        self.inner = inner_mod


class RepeatMorphism(Morphism):
    """Runtime-variable iterated composition (matrix power).

    Wraps an endomorphism f : X -> X and computes f^n at runtime,
    where n can be changed between calls. Uses repeated squaring
    for O(log n) quantale compositions.

    For an endomorphism T : S -> S under a quantale, ``T^n`` is the
    n-fold Kleisli composition. Under the product_fuzzy quantale
    with stochastic matrices, this is standard matrix power.

    Parameters
    ----------
    inner : Morphism
        An endomorphism (domain must equal codomain).
    n : int
        Initial number of repetitions (default 1). Can be changed
        later via the ``n_steps`` property.

    Raises
    ------
    TypeError
        If the inner morphism is not an endomorphism.
    ValueError
        If n < 1.

    Examples
    --------
    >>> T = morphism(S, S)
    >>> rep = RepeatMorphism(T, n=5)
    >>> rep.tensor  # computes T^5
    >>> rep.n_steps = 10
    >>> rep.tensor  # now computes T^10
    """

    def __init__(self, inner: Morphism, n: int = 1) -> None:
        if inner.domain != inner.codomain:
            raise TypeError(
                f"repeat requires an endomorphism, got "
                f"{inner.domain!r} -> {inner.codomain!r}"
            )

        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        super().__init__(
            inner.domain,
            inner.codomain,
            quantale=inner._quantale,
        )

        self._inner = inner
        self._n = n
        self._n_contract = inner.codomain.ndim

    @property
    def inner(self) -> Morphism:
        """The base endomorphism."""
        return self._inner

    @property
    def n_steps(self) -> int:
        """Number of iterated compositions."""
        return self._n

    @n_steps.setter
    def n_steps(self, value: int) -> None:
        if value < 1:
            raise ValueError(f"n_steps must be >= 1, got {value}")

        self._n = value

    @property
    def tensor(self) -> torch.Tensor:
        """Compute the n-fold composition via repeated squaring.

        Returns
        -------
        torch.Tensor
            The tensor for f^n, same shape as the inner morphism.
        """
        t = self._inner.tensor

        if self._n == 1:
            return t

        # repeated squaring: O(log n) compositions
        result = None
        base = t
        n = self._n

        while n > 0:
            if n % 2 == 1:
                if result is None:
                    result = base

                else:
                    result = self._quantale.compose(
                        result,
                        base,
                        self._n_contract,
                    )

            base = self._quantale.compose(
                base,
                base,
                self._n_contract,
            )
            n //= 2

        return result

    def module(self) -> nn.Module:
        return _RepeatModule(self._inner.module())

    def __repr__(self) -> str:
        return f"RepeatMorphism({self._inner!r}, n={self._n})"


# -- factory functions -------------------------------------------------------


def morphism(
    domain: SetObject,
    codomain: SetObject,
    init_scale: float = 0.5,
    quantale: Quantale | None = None,
) -> LatentMorphism:
    """Create a latent (learnable) morphism.

    Parameters
    ----------
    domain : SetObject
        Source object.
    codomain : SetObject
        Target object.
    init_scale : float
        Initialization scale for unconstrained parameters.
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    LatentMorphism
        A learnable morphism.
    """
    return LatentMorphism(domain, codomain, init_scale=init_scale, quantale=quantale)


def observed(
    domain: SetObject,
    codomain: SetObject,
    data: torch.Tensor,
    quantale: Quantale | None = None,
) -> ObservedMorphism:
    """Create an observed (fixed) morphism.

    Parameters
    ----------
    domain : SetObject
        Source object.
    codomain : SetObject
        Target object.
    data : torch.Tensor
        Fixed tensor.
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    ObservedMorphism
        A fixed morphism.
    """
    return ObservedMorphism(domain, codomain, data, quantale=quantale)


def identity(
    obj: SetObject,
    quantale: Quantale | None = None,
) -> ObservedMorphism:
    """Create the identity morphism on an object.

    Returns an observed morphism obj -> obj whose tensor is the
    identity: unit on the diagonal, zero elsewhere.

    Parameters
    ----------
    obj : SetObject
        The object to create an identity for.
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    ObservedMorphism
        The identity morphism.
    """
    q = quantale if quantale is not None else PRODUCT_FUZZY
    data = q.identity_tensor(obj.shape)

    return ObservedMorphism(obj, obj, data, quantale=q)
