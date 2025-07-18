import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from mycoding.fit_predict import (
    split_categorical_by_uniqueness,
    get_col_roles,
    drop_skip_cols,
    build_pipeline,
)

def test_split_categorical_by_uniqueness():
    df = pd.DataFrame({
        'col1': ['a', 'b', 'a', 'b'],
        'col2': ['x', 'y', 'z', 'w'],
        'col3': ['p'] * 4
    })
    hot_cols, cat_cols = split_categorical_by_uniqueness(df, ['col1', 'col2', 'col3'], k_threshold=2)
    assert 'col1' in hot_cols
    assert 'col2' in cat_cols
    assert 'col3' in hot_cols

def test_get_col_roles_keys():
    dct_cols = get_col_roles()
    for key in ['num', 'cat', 'hot', 'tgt', 'dat']:
        assert key in dct_cols

def test_drop_skip_cols():
    dct_cols = {'num': ['a', 'b'], 'cat': ['c']}
    skip_cols = ['b']
    cleaned = drop_skip_cols(skip_cols, dct_cols)
    assert 'b' not in cleaned['num']
    assert cleaned['cat'] == ['c']

def test_build_pipeline_structure():
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['feature1'])
    ])
    model = LogisticRegression()
    pipeline = build_pipeline(model, preprocessor)
    assert 'preprocessor' in pipeline.named_steps
    assert 'classifier' in pipeline.named_steps


