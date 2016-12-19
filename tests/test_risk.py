from ceam_public_health.util.risk import naturally_sort_df

def test_naturally_sort_df():
    df = pd.DataFrame({'simulant_id': [x for x in range(0, 5)], 'cat2': 5 * [.5], 'cat1': 5*[.1], 'cat3': 5* [.4] })
    df, categories = naturally_sort_df(df)
    assert df.columns.tolist() == ['cat1', 'cat2', 'cat3'], "naturally_sort_df should naturally sort (i.e. cat1, cat2, etc.) category columns"
    assert categories == ['cat1', 'cat2', 'cat3'], "naturally_sort_df should return all of the categories for a given categorical risk factor"


