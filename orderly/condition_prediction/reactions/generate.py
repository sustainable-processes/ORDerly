import pathlib
import multiprocessing
import typing
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

import param_sharing.reactions.fingerprint
import param_sharing.reactions.get


def generate_data(
    cleaned_data_path: pathlib.Path = pathlib.Path("data/ORD_USPTO/cleaned_data.pkl"),
    rxn_classes_path: pathlib.Path = pathlib.Path("data/ORD_USPTO/classified_rxn.smi"),
    diff_fp_path: typing.Optional[pathlib.Path] = pathlib.Path(
        "data/ORD_USPTO/USPTO_rxn_diff_fp.pkl"
    ),
    product_fp_path: typing.Optional[pathlib.Path] = pathlib.Path(
        "data/ORD_USPTO/USPTO_product_fp.pkl"
    ),
    reactant_fp_path: typing.Optional[pathlib.Path] = pathlib.Path(
        "data/ORD_USPTO/USPTO_reactant_fp.pkl"
    ),
    radius: int = 3,
    nBits: int = 512,
):
    df = param_sharing.reactions.get.get_reaction_df(
        cleaned_data_path=cleaned_data_path,
        rxn_classes_path=rxn_classes_path,
    )

    # test_df = df['rxn_class'].str.rsplit(';', expand=True)
    # 2.5% of reactions have been assigned 2 reaction classes. 3 or 4 reaction classes is very rare.

    # calculate rxn difference fp
    # converting one 500k by 2k list to array takes roughly 15s, so the whole thing should take about 2-3 min
    # need to split into different cells for memory purposes

    num_cores = multiprocessing.cpu_count()
    inputs = tqdm(df["product_0"])
    p0 = Parallel(n_jobs=num_cores)(
        delayed(param_sharing.reactions.fingerprint.calc_fp_individual)(
            i, radius, nBits
        )
        for i in inputs
    )

    inputs = tqdm(df["product_1"])
    p1 = Parallel(n_jobs=num_cores)(
        delayed(param_sharing.reactions.fingerprint.calc_fp_individual)(
            i, radius, nBits
        )
        for i in inputs
    )

    inputs = tqdm(df["product_2"])
    p2 = Parallel(n_jobs=num_cores)(
        delayed(param_sharing.reactions.fingerprint.calc_fp_individual)(
            i, radius, nBits
        )
        for i in inputs
    )

    inputs = tqdm(df["product_3"])
    p3 = Parallel(n_jobs=num_cores)(
        delayed(param_sharing.reactions.fingerprint.calc_fp_individual)(
            i, radius, nBits
        )
        for i in inputs
    )

    ar_p0 = np.array(p0)
    ar_p1 = np.array(p1)
    ar_p2 = np.array(p2)
    ar_p3 = np.array(p3)

    product_fp = ar_p0 + ar_p1 + ar_p2 + ar_p3

    del ar_p0, ar_p1, ar_p2, ar_p3
    del p0, p1, p2, p3

    num_cores = multiprocessing.cpu_count()
    inputs = tqdm(df["reactant_0"])
    r0 = Parallel(n_jobs=num_cores)(
        delayed(param_sharing.reactions.fingerprint.calc_fp_individual)(
            i, radius, nBits
        )
        for i in inputs
    )

    inputs = tqdm(df["reactant_1"])
    r1 = Parallel(n_jobs=num_cores)(
        delayed(param_sharing.reactions.fingerprint.calc_fp_individual)(
            i, radius, nBits
        )
        for i in inputs
    )

    inputs = tqdm(df["reactant_2"])
    r2 = Parallel(n_jobs=num_cores)(
        delayed(param_sharing.reactions.fingerprint.calc_fp_individual)(
            i, radius, nBits
        )
        for i in inputs
    )

    inputs = tqdm(df["reactant_3"])
    r3 = Parallel(n_jobs=num_cores)(
        delayed(param_sharing.reactions.fingerprint.calc_fp_individual)(
            i, radius, nBits
        )
        for i in inputs
    )

    ar_r0 = np.array(r0)
    ar_r1 = np.array(r1)
    ar_r2 = np.array(r2)
    ar_r3 = np.array(r3)

    reactant_fp = ar_r0 + ar_r1 + ar_r2 + ar_r3
    del ar_r0, ar_r1, ar_r2, ar_r3
    del r0, r1, r2, r3

    rxn_diff_fp = product_fp - reactant_fp

    # save to pickle
    if diff_fp_path is not None:
        np.save(diff_fp_path, rxn_diff_fp)
    if product_fp_path is not None:
        np.save(product_fp_path, product_fp)
    if reactant_fp_path is not None:
        np.save(reactant_fp_path, reactant_fp)
    return rxn_diff_fp, product_fp


if __name__ == "__main__":
    generate_data(
        cleaned_data_path=pathlib.Path("data/ORD_USPTO/cleaned_data.pkl"),
        rxn_classes_path=pathlib.Path("data/ORD_USPTO/classified_rxn.smi"),
        diff_fp_path=pathlib.Path("data/ORD_USPTO/USPTO_rxn_diff_fp.pkl"),
        product_fp_path=pathlib.Path("data/ORD_USPTO/USPTO_product_fp.pkl"),
        reactant_fp_path=pathlib.Path("data/ORD_USPTO/USPTO_reactant_fp.pkl"),
        radius=3,
        nBits=512,
    )
