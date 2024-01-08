enzyme_type = {
    1: "Oxidoreductases",
    2: "Transferases",
    3: "Hydrolases",
    4: "Lyases",
    5: "Isomerases",
    6: "Ligases",
    7: "Translocases"
 }

# this is the 3rd number for oxidoreductases
oxidoreductase_acceptor = {
    1: "NAD+ or NADP+",
    2: "cytochrome",
    3: "oxygen",
    4: "disulfide",
    5: "quinone",
    6: "nitrogenous group",
    7: "iron-sulfur protein",
    8: "flavin group",
    9: "copper protein",
}

sub_type = {
    "Oxidoreductases": {
        1: "CH-OH group of donors",
        2: "aldehyde or oxo group of donors",
        3: "CH-CH group of donors",
        4: "CH-NH2 group of donors",
        5: "CH-NH group of donors",
        6: "NADH or NADPH",
        7: "other nitrogenous compounds as donors",
        8: "sulfur group of donors",
        9: "heme group of donors",
        10: "diphenols and related substances as donors",
        11: "peroxide as acceptor",
        12: "hydrogen as donor",
        13: "oxygenases",
        14: "paired donors, with molecular oxygen",
        15: "superoxide as acceptor",
        16: "Oxidizing metal ions",
        17: "CH or CH2 groups",
        18: "iron-sulfur proteins as donors",
        19: "reduced flavodoxin as donor",
        20: "phosphorus or arsenic in donors",
        21: "Catalysing the reaction X-H + Y-H = X-Y",
        22: "halogen in donors",
        23: "Reducing C-O-C group as acceptor",
    },

    "Transferases": {
        1: "one-carbon groups",
        2: "aldehyde or ketonic groups",
        3: "Acyltransferases",
        4: "Glycosyltransferases",
        5: "alkyl or aryl groups, other than methyl groups",
        6: "nitrogenous groups",
        7: "phosphorus-containing groups",
        8: "sulfur-containing groups",
        9: "selenium-containing groups",
        10: "molybdenum- or tungsten-containing groups",
    },

    "Hydrolases": {
        1: "ester bonds",
        2: "Glycosylases",
        3: "ether bonds",
        4: "peptide bonds (peptidases)",
        5: "carbon-nitrogen bonds, other than peptide bonds",
        6: "acid anhydrides",
        7: "carbon-carbon bonds",
        8: "halide bonds",
        9: "phosphorus-nitrogen bonds",
        10: "sulfur-nitrogen bonds",
        11: "carbon-phosphorus bonds",
        12: "sulfur-sulfur bonds",
        13: "carbon-sulfur bonds",
    },

    "Lyases": {
        1: "Carbon-carbon",
        2: "Carbon-oxygen",
        3: "Carbon-nitrogen",
        4: "Carbon-sulfur",
        5: "Carbon-halide",
        6: "Phosphorus-oxygen",
        7: "Carbon-phosphorus",
        8: "Nitrogen-oxygen",
    },

    "Isomerases": {
        1: "Racemases and epimerases",
        2: "cis-trans-Isomerases",
        3: "Intramolecular oxidoreductases",
        4: "Intramolecular transferases",
        5: "Intramolecular lyases",
        6: "Isomerases altering macromolecular conformation",
    },

    "Ligases": {
        1: "carbon-oxygen bonds",
        2: "carbon-sulfur bonds",
        3: "carbon-nitrogen bonds",
        4: "carbon-carbon bonds",
        5: "phosphoric-ester bonds",
        6: "nitrogen—metal bonds",
        7: "nitrogen-nitrogen bonds",
    },

    "Translocases": {
        1: "hydrons",
        2: "inorganic cations",
        3: "inorganic anions and their chelates",
        4: "amino acids and peptides",
        5: "carbohydrates and their derivatives",
        6: "other compounds",
    },
}


def add_enzyme_types(df):
    """ Using a column named "ec_num" (EC number), get the enzyme type and sub type
    """
    df = df.copy()
    df['enzyme_type'] = df.ec_num.apply(get_enzyme_type)
    df['enzyme_type_sub'] = df.ec_num.apply(get_enzyme_type_sub)
    df = get_redox_acceptor(df)
    return df


def get_enzyme_type(ec_num):
    return enzyme_type[int(ec_num.split('.')[0])]


def get_enzyme_type_sub(ec_num):
    enzyme_type_sub = sub_type[enzyme_type[int(ec_num.split('.')[0])]].get(int(ec_num.split('.')[1]), 'other')
    return enzyme_type_sub


def get_redox_acceptor(df):
    df.loc[df.enzyme_type == "Oxidoreductases", "acceptor"] = df.loc[
        df.enzyme_type == "Oxidoreductases"].ec_num.apply(
        lambda x: oxidoreductase_acceptor.get(int(x.split('.')[2]), "other"))
    return df
