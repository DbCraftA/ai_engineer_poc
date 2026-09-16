"""Section 1.8 — dtype, précision de mantisse et plage dynamique.

Deux phénomènes distincts se cachent derrière « réduire la précision » :

- la **mantisse** fixe l'erreur d'arrondi (FP32 : 24 bits, FP16 : 11 bits, BF16 : 8 bits) ;
- l'**exposant** fixe la plage représentable (FP16 : +/-65504, BF16 : celle de FP32).

Les métriques d'erreur mesurent le premier, les fonctions de plage (`representable_range`,
`count_overflows`) le second. Un débordement produit volontairement `inf` : la valeur est perdue en
totalité, donc l'erreur est infinie. Ce module ne clampe jamais un débordement, mais il ne renvoie
jamais `nan` non plus, car `nan` rendrait silencieusement fausse toute comparaison en aval.
"""

import torch

# Dtypes dont le coût mémoire par élément est défini dans ce dépôt. Tout le reste (complexes,
# tenseurs quantifiés, dtypes exotiques) doit échouer explicitement plutôt que renvoyer une valeur
# que personne n'a validée.
_SUPPORTED_DTYPES: tuple[torch.dtype, ...] = tuple(
    dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    )
    if dtype is not None
)


def _require_supported(dtype: torch.dtype) -> None:
    if not isinstance(dtype, torch.dtype) or dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"dtype non supporté par inference_lab.tensors.dtypes : {dtype!r}")


def _require_floating_point(dtype: torch.dtype) -> None:
    _require_supported(dtype)
    if not dtype.is_floating_point:
        raise TypeError(f"dtype flottant attendu, reçu {dtype!r}")


def bytes_per_element(dtype: torch.dtype) -> int:
    """Nombre d'octets occupés par UN élément de ce `dtype`."""
    _require_supported(dtype)
    return dtype.itemsize


def representable_range(dtype: torch.dtype) -> tuple[float, float]:
    """Bornes finies représentables par `dtype`, sous la forme `(min, max)`.

    C'est l'exposant, pas la mantisse, qui fixe ces bornes : FP16 s'arrête à 65504 alors que BF16
    partage la plage de FP32 (~3.4e38).
    """
    _require_floating_point(dtype)
    info = torch.finfo(dtype)
    return float(info.min), float(info.max)


def roundtrip(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Aller-retour `tensor -> dtype -> dtype d'origine`, pour observer ce que le cast a perdu."""
    initial_type = tensor.dtype
    return tensor.to(dtype).to(initial_type)


def count_overflows(tensor: torch.Tensor, dtype: torch.dtype) -> int:
    """Nombre d'éléments finis que le cast vers `dtype` rend non finis (débordement d'exposant).

    Les éléments déjà non finis en entrée ne sont pas comptés : on mesure ce que *le cast* détruit,
    pas ce que l'appelant a fourni.
    """
    _require_floating_point(dtype)
    finite_before = tensor.isfinite()
    finite_after = tensor.to(dtype).isfinite()
    return int((finite_before & ~finite_after).sum().item())


def max_absolute_error(tensor: torch.Tensor, dtype: torch.dtype) -> float:
    """Erreur absolue maximale d'un aller-retour vers `dtype`.

    Renvoie `inf` si un élément déborde de la plage de `dtype` : le cast a perdu la valeur, l'erreur
    est bien infinie. Seuls les éléments finis en entrée sont considérés, pour ne jamais renvoyer
    `nan` (`inf - inf`).
    """
    _require_floating_point(dtype)
    finite = tensor.isfinite()
    if not bool(finite.any()):
        return 0.0
    error = (tensor - roundtrip(tensor, dtype)).abs()
    return float(error[finite].max().item())


def max_relative_error(tensor: torch.Tensor, dtype: torch.dtype) -> float:
    """Erreur relative maximale d'un aller-retour vers `dtype`, sur les éléments finis et non nuls.

    Indispensable pour voir le flush-to-zero : une valeur de 1e-8 écrasée à 0 en FP16 ne coûte que
    1e-8 en erreur absolue (donc « négligeable ») alors qu'elle est perdue à 100 %.
    """
    _require_floating_point(dtype)
    eligible = tensor.isfinite() & (tensor != 0)
    if not bool(eligible.any()):
        return 0.0
    error = (tensor - roundtrip(tensor, dtype)).abs()[eligible]
    return float((error / tensor.abs()[eligible]).max().item())



def tensor_memory_bytes(tensor: torch.Tensor):
    return tensor.numel() * bytes_per_element(tensor.dtype)