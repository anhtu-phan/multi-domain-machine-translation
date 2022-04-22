import pandas as pd

dataset_path = './datasets/de-en/mixed'


def build_data(data_type, src_ext, trg_ext):
    src_data = []
    trg_data = []

    with open(f"{dataset_path}/{data_type}.{src_ext}", "rt", encoding="utf8") as f:
        for line in f:
            src_data.append(line)

    with open(f"{dataset_path}/{data_type}.{trg_ext}", "rt", encoding="utf8") as f:
        for line in f:
            trg_data.append(line)

    data = {"source": src_data, "target": trg_data}
    df = pd.DataFrame(data)
    df.to_csv(f"{dataset_path}/{data_type}.tsv", sep="\t", index=False)


build_data("train", "en", "de")
build_data("valid", "en", "de")
build_data("test", "en", "de")
