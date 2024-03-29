from typing import List, Dict, Any, Optional, Iterable, TypeVar, NewType, Union

REPEATEDCOMPOSITECONTAINER = TypeVar(
    "REPEATEDCOMPOSITECONTAINER", bound=Iterable[Any]
)  # protobuf uses a different type for the repeat composite container for each OS so we need a generic type that is not using the true type

MOLECULE_IDENTIFIER = str  # NewType('MOLECULE_IDENTIFIER', str)
INVALID_IDENTIFIER = str  # NewType('INVALID_IDENTIFIER', str)

SMILES = str  # NewType('SMILES', str)
CANON_SMILES = (
    str  # NewType('CANON_SMILES', SMILES)  # This is for SMILES canonicalised by RDKit
)

MANUAL_REPLACEMENTS_DICT = Dict[
    MOLECULE_IDENTIFIER | INVALID_IDENTIFIER, Optional[SMILES | CANON_SMILES]
]

RXN_STR = NewType("RXN_STR", str)

REAGENT = Union[CANON_SMILES, SMILES, MOLECULE_IDENTIFIER]
CANON_REAGENT = CANON_SMILES
REAGENTS = List[REAGENT]
CANON_REAGENTS = List[CANON_REAGENT]

REACTANT = Union[CANON_SMILES, SMILES, MOLECULE_IDENTIFIER]
CANON_REACTANT = CANON_SMILES
REACTANTS = List[REACTANT]
CANON_REACTANTS = List[CANON_REACTANT]

CATALYST = Union[CANON_SMILES, SMILES, MOLECULE_IDENTIFIER]
CANON_CATALYST = CANON_SMILES
CATALYSTS = List[CATALYST]
CANON_CATALYSTS = List[CANON_CATALYST]

PRODUCT = Union[CANON_SMILES, SMILES, MOLECULE_IDENTIFIER]
CANON_PRODUCT = CANON_SMILES
PRODUCTS = List[PRODUCT]
CANON_PRODUCTS = List[CANON_PRODUCT]

SOLVENT = Union[CANON_SMILES, SMILES, MOLECULE_IDENTIFIER]
CANON_SOLVENT = CANON_SMILES
SOLVENTS = List[SOLVENT]
CANON_SOLVENTS = List[CANON_SOLVENT]

AGENT = Union[CANON_SMILES, SMILES, MOLECULE_IDENTIFIER]
CANON_AGENT = CANON_SMILES
AGENTS = List[AGENT]
CANON_AGENTS = List[CANON_AGENT]

YIELD = NewType("YIELD", float)
YIELDS = List[Optional[YIELD]]

TEMPERATURE_CELCIUS = NewType("TEMPERATURE_CELCIUS", float)
TEMPERATURES_CELCIUS = List[Optional[TEMPERATURE_CELCIUS]]

RXN_TIME = NewType("RXN_TIME", float)  # hours
