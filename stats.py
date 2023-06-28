import argparse
import pathlib
import typing as tp

import numpy as np
import pandas as pd

import categories


category_to_main_category = {value: key for key, sublist in categories.categories.items() for value in sublist}
subcategories2categories = {key: category_to_main_category[value[0]] for key, value in categories.subcategories.items()}

def calculate_accuracy_from_directory(dirpath: str) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert pathlib.Path(dirpath).exists()
    filepaths = [str(x) for x in pathlib.Path(dirpath).glob('*.jsonl')]
    assert len(filepaths) == 57
    res = {}
    for each_filepath in filepaths:
        df = pd.read_json(each_filepath, lines=True)
        df.columns = ['prompt', 'label', 'preds']
        cors = []
        for idx, row in df.iterrows():
            preds = row['preds']
            best_idx = np.argmax(list(preds.values()))
            y_pred = list(preds.keys())[best_idx]
            y_true = row['label']
            y_pred = y_pred.strip()
            y_true = y_true.strip()
            cors.append(y_true == y_pred)
        acc = np.mean(cors)
        res[pathlib.Path(each_filepath).stem] = acc * 100
    
    df = pd.DataFrame({pathlib.Path(dirpath).stem: res}).reset_index()
    df = df.rename(columns={'index': 'subcategory'})
    subcategories_df = df.copy()
    
    df = subcategories_df.copy()
    df['subcategory'] = df['subcategory'].map(subcategories2categories)
    df = df.rename(columns={'subcategory': 'category'})
    df = df.groupby('category').mean().reset_index()
    categories_df = df.copy()
    
    total_df = pd.DataFrame({pathlib.Path(dirpath).stem: [categories_df[pathlib.Path(dirpath).stem].mean()]})
    
    assert subcategories_df.shape == (57, 2)
    assert categories_df.shape == (4, 2)
    assert total_df.shape == (1, 1)
    return (subcategories_df, categories_df, total_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirpath", type=str)
    args = parser.parse_args()    
    print(f'Calculating accuracy for dir: {args.dirpath}')
    (subcategories_df, categories_df, total_df) = calculate_accuracy_from_directory(dirpath=args.dirpath)
    print("-=-=-=- Subcategories -=-=-=-")
    print(subcategories_df.to_string())
    print("-=-=-=- Categories -=-=-=-")
    print(categories_df.to_string())
    print("-=-=-=- Total -=-=-=-")
    print(total_df.to_string())
    print("-=-=-=-=-=-=-=-=-=-=-")
