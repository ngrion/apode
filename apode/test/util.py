import pandas as pd
import numpy as np


# flatten = lambda l: [item for sublist in l for item in sublist]
def flatten(long_list):
    flat = []
    for sublist in long_list:
        for item in sublist:
            flat.append(item)
    return flat


def joinpar(x, y):
    if isinstance(x, list):
        return flatten([[x[0]], [y], x[1:]])
    else:
        return [x, y]


# Evaluar un listado de métodos
def test_measures(df, type):
    if type == "poverty":
        return test_measures_poverty(df)
    elif type == "ineq":
        return test_measures_ineq(df)
    elif type == "welfare":
        return test_measures_welfare(df)
    elif type == "polar":
        return test_measures_polar(df)
    elif type == "conc":
        return test_measures_conc(df)
    else:
        raise ValueError("La categoria no existe.")


def test_measures_poverty(dr2):
    pline = 50  # Poverty line
    mlist_p = [
        "fgt0",
        "fgt1",
        "fgt2",
        ["fgt", 1.5],
        "sen",
        "sst",
        "watts",
        ["cuh", 0],
        ["cuh", 0.5],
        "takayama",
        "kakwani",
        "thon",
        ["bd", 2],
        "hagenaars",
        ["chakravarty", 0.5],
    ]
    mlist_p2 = [joinpar(x, pline) for x in mlist_p]
    table = []
    for elem in mlist_p2:
        table.append(dr2.poverty(elem[0], *elem[1:]))
    df_outp = pd.DataFrame(mlist_p2, columns=["method", "pline", "par"])
    df_outp["poverty_measure"] = table
    return df_outp


def test_measures_ineq(dr2):
    list_i = [
        "rr",
        "dmr",
        "cv",
        "dslog",
        "gini",
        "merhan",
        "piesch",
        "bonferroni",
        ["kolm", 0.5],
        ["ratio", 0.05],
        ["ratio", 0.2],
        ["entropy", 0],
        ["entropy", 1],
        ["entropy", 2],
        ["atkinson", 0.5],
        ["atkinson", 1],
        ["atkinson", 2],
    ]
    list_i = [[elem] if not isinstance(elem, list)
              else elem for elem in list_i]
    table = []
    for elem in list_i:
        table.append(dr2.ineq(*elem))
    dz_i = pd.DataFrame(list_i, columns=["method", "par"])
    dz_i["ineq_measure"] = table
    return dz_i


def test_measures_welfare(dr2):
    list_w = [
        "utilitarian",
        "rawlsian",
        "sen",
        "theill",
        "theilt",
        ["isoelastic", 0],
        ["isoelastic", 1],
        ["isoelastic", 2],
        ["isoelastic", np.Inf],
    ]
    list_w = [[elem] if not isinstance(elem, list)
              else elem for elem in list_w]
    table = []
    for elem in list_w:
        table.append(dr2.welfare(*elem))
    dz_w = pd.DataFrame(list_w, columns=["method", "par"])
    dz_w["welfare_measure"] = table
    return dz_w


def test_measures_polar(dr2):
    list_pz = ["er", "wlf"]
    table = []
    for elem in list_pz:
        table.append(dr2.polar(elem))
    dz_pz = pd.DataFrame(list_pz, columns=["method"])
    dz_pz["polarization_measure"] = table
    return dz_pz


def test_measures_conc(dr2):
    list_c = ["hhi", "hhin", "rosenbluth", ["cr", 1], ["cr", 5]]
    list_c = [[elem] if not isinstance(elem, list)
              else elem for elem in list_c]
    table = []
    for elem in list_c:
        table.append(dr2.conc(*elem))
    dz_c = pd.DataFrame(list_c, columns=["method", "par"])
    dz_c["concentration_measure"] = table
    return dz_c
