import pandas as pd

def real_estate():
    df=pd.read_csv('Datasets/REAL_ESTATE.csv')
    df = df.drop(df.columns[:2], axis=1)
    df = df.map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    df = df.astype(float)
    target = df[df.columns[-1]]
    data = df[df.columns[:-1]]
    data = data.to_numpy()
    target = target.to_numpy()
    return data, target, data.shape[1]

def concrete():
    df = pd.read_csv("Datasets/CONCRETE_COMPRESSIVE_STRENGTH.csv")
    target = df[df.columns[-1]]
    data = df[df.columns[:-1]]
    # remove is na values
    data = data.dropna()
    target = target.dropna()
    data = data.to_numpy()
    target = target.to_numpy()
    return data, target, data.shape[1]

def energy():
    df = pd.read_csv("Datasets/ENERGY_EFFICIENCY.csv")
    df = df.map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    df = df.astype(float)
    df = df.drop(df.columns[-1],axis=1)
    data = df[df.columns[:-1]]
    target = df[df.columns[-1]]
    data = data.to_numpy()
    target = target.to_numpy()
    return data, target, data.shape[1]


def yacht_hydrodynamics():
    df = pd.read_csv("Datasets/YACHT.csv")
    df = df.map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    df = df.astype(float)
    data = df[df.columns[:-1]]
    target = df[df.columns[-1]]
    data = data.to_numpy()
    target = target.to_numpy()
    return data, target, data.shape[1]

def superconductivity():
    df = pd.read_csv("Datasets/SUPERCONDUCTIVITY.csv")
    df = df.map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    df = df.astype(float)
    data = df[df.columns[:-1]]
    target = df[df.columns[-1]]
    data = data.to_numpy()
    target = target.to_numpy()
    return data, target, data.shape[1]

def get_dataset(dataset_name):
    """Get dataset based on name from config"""

    if dataset_name == "superconductivity":
        return superconductivity()
    elif dataset_name == "concrete":
        return concrete()
    elif dataset_name == "real_estate":
        return real_estate()
    elif dataset_name == "energy":
        return energy()
    elif dataset_name == "yacht":
        return yacht_hydrodynamics()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

