"""Day convolution for V-enriched presheaves.

Given a monoidal structure (C, ⊗, I) on a V-enriched category C, the
Day convolution defines a monoidal structure on the presheaf category
[C^op, V]. For two presheaves F, G: C^op → V, their Day convolution
is:

    (F ⊛ G)(C) = ∫^{A,B} C(C, A ⊗ B) ⊗ F(A) ⊗ G(B)

In our finite setting, this becomes a computable coend (join over
all pairs (A, B) such that A ⊗ B can map to C).

The unit for Day convolution is the representable presheaf at the
monoidal unit: y(I)(C) = C(C, I).

This module provides:

    day_convolution()          — compute F ⊛ G
    day_unit()                 — the unit presheaf for ⊛
    day_convolution_profunctors() — Day convolution on profunctors
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from quivers.core.objects import SetObject
from quivers.core.quantales import PRODUCT_FUZZY, Quantale
from quivers.categorical.monoidal import MonoidalStructure
from quivers.enriched.profunctors import Profunctor


def day_convolution(
    f_values: torch.Tensor,
    g_values: torch.Tensor,
    objects: Sequence[SetObject],
    monoidal: MonoidalStructure,
    product_table: dict[tuple[int, int], tuple[int, torch.Tensor]] | None = None,
    quantale: Quantale | None = None,
) -> torch.Tensor:
    """Compute the Day convolution (F ⊛ G)(C) for each object C.

    For finite discrete categories (no non-identity morphisms),
    the Day convolution simplifies to:

        (F ⊛ G)(C) = ⋁_{A,B : A⊗B=C} F(A) ⊗ G(B)

    For general finite categories with hom-data:

        (F ⊛ G)(C) = ⋁_{A,B} C(C, A⊗B) ⊗ F(A) ⊗ G(B)

    Parameters
    ----------
    f_values : torch.Tensor
        1D tensor of shape (n,) representing F evaluated at each
        object: F(objects[i]) = f_values[i].
    g_values : torch.Tensor
        1D tensor of shape (n,) representing G evaluated at each
        object: G(objects[i]) = g_values[i].
    objects : Sequence[SetObject]
        The objects of the finite category (indexed 0..n-1).
    monoidal : MonoidalStructure
        The monoidal structure providing ⊗ and I.
    product_table : dict or None
        Optional precomputed product table mapping (i, j) to
        (k, hom_tensor) where objects[i] ⊗ objects[j] ≅ objects[k]
        and hom_tensor is C(objects[k], objects[i] ⊗ objects[j]).
        If None, uses identity hom (discrete category).
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    torch.Tensor
        1D tensor of shape (n,) with (F ⊛ G)(objects[i]) at index i.
    """
    q = quantale if quantale is not None else PRODUCT_FUZZY
    n = len(objects)

    if f_values.shape != (n,) or g_values.shape != (n,):
        raise ValueError(
            f"f_values and g_values must have shape ({n},), got "
            f"{f_values.shape} and {g_values.shape}"
        )

    result = torch.full((n,), q.zero)

    for i in range(n):
        for j in range(n):
            f_i = f_values[i]
            g_j = g_values[j]

            # compute f(A) ⊗ g(B)
            fg = q.tensor_op(f_i.unsqueeze(0), g_j.unsqueeze(0)).squeeze(0)

            if product_table is not None:
                if (i, j) not in product_table:
                    continue

                k, hom = product_table[(i, j)]

                # C(C_k, A_i ⊗ B_j) ⊗ F(A_i) ⊗ G(B_j)
                contribution = q.tensor_op(hom, fg)

                # join into result[k]
                current = result[k].unsqueeze(0)
                new_val = q.join(
                    torch.stack([current, contribution.unsqueeze(0)]),
                    dim=0,
                )
                result[k] = new_val.squeeze(0)

            else:
                # discrete case: A_i ⊗ A_j must map to some A_k
                # for FinSets with cartesian monoidal, this only
                # works when objects are closed under ⊗
                prod = monoidal.product(objects[i], objects[j])

                # find matching object
                for k in range(n):
                    if objects[k].shape == prod.shape:
                        current = result[k].unsqueeze(0)
                        new_val = q.join(
                            torch.stack([current, fg.unsqueeze(0)]),
                            dim=0,
                        )
                        result[k] = new_val.squeeze(0)

                        break

    return result


def day_unit(
    objects: Sequence[SetObject],
    unit_index: int,
    quantale: Quantale | None = None,
) -> torch.Tensor:
    """Create the unit presheaf for Day convolution.

    The unit is the representable presheaf y(I) at the monoidal
    unit I. In the discrete case: y(I)(C) = I if C = I, else ⊥.

    Parameters
    ----------
    objects : Sequence[SetObject]
        The objects of the finite category.
    unit_index : int
        The index of the monoidal unit in the objects list.
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    torch.Tensor
        1D tensor of shape (n,) representing the unit presheaf.
    """
    q = quantale if quantale is not None else PRODUCT_FUZZY
    n = len(objects)
    result = torch.full((n,), q.zero)
    result[unit_index] = q.unit

    return result


def day_convolution_profunctors(
    p: Profunctor,
    q_prof: Profunctor,
    monoidal: MonoidalStructure,
    quantale: Quantale | None = None,
) -> Profunctor:
    """Day convolution of two profunctors.

    Given profunctors P: A ↛ B and Q: C ↛ D, the Day convolution
    (with respect to the monoidal product) produces:

        (P ⊛ Q): A⊗C ↛ B⊗D

    with tensor:

        (P ⊛ Q)(ac, bd) = P(a, b) ⊗ Q(c, d)

    This is the external tensor product of profunctors, which
    extends the monoidal structure from C to Prof(C).

    Parameters
    ----------
    p : Profunctor
        Left profunctor P: A ↛ B.
    q_prof : Profunctor
        Right profunctor Q: C ↛ D.
    monoidal : MonoidalStructure
        The monoidal structure providing ⊗.
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    Profunctor
        The Day convolution P ⊛ Q: A⊗C ↛ B⊗D.
    """
    qnt = quantale if quantale is not None else PRODUCT_FUZZY

    contra = monoidal.product(p.contra, q_prof.contra)
    co = monoidal.product(p.co, q_prof.co)

    # compute outer tensor product
    pt = p.tensor
    qt = q_prof.tensor

    n_p = pt.ndim
    n_q = qt.ndim

    # expand: P(*a, *b, 1..1) ⊗ Q(1..1, *c, *d)
    pt_exp = pt.reshape(*pt.shape, *([1] * n_q))
    qt_exp = qt.reshape(*([1] * n_p), *qt.shape)

    outer = qnt.tensor_op(pt_exp, qt_exp)

    # current layout: [a_dims, b_dims, c_dims, d_dims]
    # target layout:  [a_dims, c_dims, b_dims, d_dims]
    n_a = p.contra.ndim
    n_b = p.co.ndim
    n_c = q_prof.contra.ndim
    n_d = q_prof.co.ndim

    a_dims = list(range(n_a))
    b_dims = list(range(n_a, n_a + n_b))
    c_dims = list(range(n_a + n_b, n_a + n_b + n_c))
    d_dims = list(range(n_a + n_b + n_c, n_a + n_b + n_c + n_d))

    perm = a_dims + c_dims + b_dims + d_dims
    result = outer.permute(*perm)

    return Profunctor(contra=contra, co=co, tensor=result, quantale=qnt)
