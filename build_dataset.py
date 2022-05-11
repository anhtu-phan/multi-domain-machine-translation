import pandas as pd


def build_data(dataset_path, src_ext, trg_ext, domain=None):
    src_data = []
    trg_data = []

    with open(f"{dataset_path}.{src_ext}", "rt", encoding="utf8") as f:
        for line in f:
            src_data.append(line)

    with open(f"{dataset_path}.{trg_ext}", "rt", encoding="utf8") as f:
        for line in f:
            trg_data.append(line)
    if domain is not None:
        data = {"src": src_data, "trg": trg_data, "domain": [domain]*len(src_data)}
    else:
        data = {"src": src_data, "trg": trg_data}
    df = pd.DataFrame(data)
    # df.to_csv(f"{dataset_path}.tsv", sep="\t", index=False)
    return df
