def get_classified_rxn_data_mask(data_df):
    return ~data_df.rxn_class.str.contains("0.0")
