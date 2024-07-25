import os

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from dance.modules.base import BaseClassificationMethod
from dance.transforms import Compose, SetConfig, WeightedFeaturePCA
from dance.typing import LogLevel, Optional
from dance.utils.deprecate import deprecated

from dance.transforms.base import BaseTransform
# Steps for the SVM preprocessing pipeline from the paper mentioned in dance readme.

class CombinedCellGeneFilter(BaseTransform):
    """
    Combined filter for genes and cells based on various criteria.
    
    Parameters
    ----------
    mad_threshold : float
        Threshold for filtering cells based on number of detected genes.
    min_cells_per_population : int
        Minimum number of cells required for a cell population to be retained.
    """
    DISPLAY_ATTRS = ("mad_threshold", "min_cells_per_population")

    def __init__(self, mad_threshold: float = 3, min_cells_per_population: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.mad_threshold = mad_threshold
        self.min_cells_per_population = min_cells_per_population

    def __call__(self, data):
        # Get the data as a numpy array
        X = data.get_x()
        y = data.get_y()  # Assuming this returns labels as a numpy array
        cell_mask = np.ones(X.shape[0], dtype=bool)
        
        # Filter genes with zero counts
        gene_counts = X.sum(axis=0)
        non_zero_genes = gene_counts > 0
        X = X[:, non_zero_genes]

        # Filter cells by detected genes
        detected_genes = (X > 0).sum(axis=1)
        median_detected_genes = np.median(detected_genes)
        mad = np.median(np.abs(detected_genes - median_detected_genes))
        threshold = median_detected_genes - self.mad_threshold * mad
        cell_mask = cell_mask & (detected_genes >= threshold)
        X = X[cell_mask]
        y = y[cell_mask]

        # Filter cell populations
        population_counts = y.sum(axis=0)
        large_populations = population_counts >= self.min_cells_per_population
        cell_mask = cell_mask & (y[:, large_populations].any(axis=1))
        X = X[cell_mask]
        y = y[cell_mask]

        data.data.obsm[self.out] = X
        #data.data.varm[self.out] = X.T
        data.data.obsm['cell_type'] = y

        return data
    
class SVM(BaseClassificationMethod):
    """The SVM cell-type classification model.

    Parameters
    ----------
    args : argparse.Namespace
        A Namespace contains arguments of SVM. See parser help document for more info.
    prj_path: str
        project path

    """

    def __init__(self, args, prj_path="./", random_state: Optional[int] = None):
        self.args = args
        self.random_state = random_state
        self._mdl = SVC(random_state=random_state, probability=True)
        
    @staticmethod
    def preprocessing_pipeline(n_components: int = 400, log_level: LogLevel = "INFO"):
        return Compose(
            SetConfig({
                "label_channel": "cell_type"
            }),
            CombinedCellGeneFilter(mad_threshold=3, min_cells_per_population=10),
            SetConfig({
                "feature_channel": "CombinedCellGeneFilter"
            }),
            log_level=log_level,
        )

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Train the classifier.

        Parameters
        ----------
        x
            Training cell features.
        y
            Training labels.

        """
        self._mdl.fit(x, y)

    def predict(self, x: np.ndarray):
        """Predict cell labels.

        Parameters
        ----------
        x
            Samples to be predicted (samplex x features).

        Returns
        -------
        y
            Predicted labels of the input samples.

        """
        return self._mdl.predict(x)

    @deprecated
    def save(self, num, pred):
        """Save the predictions.

        Parameters
        ----------
        num: int
            test file name
        pred: dict
            prediction labels

        """
        label_map = pd.read_excel(self.prj_path / "data" / "celltype2subtype.xlsx", sheet_name=self.args.species,
                                  header=0, names=["species", "old_type", "new_type", "new_subtype"])

        save_path = self.prj_path / self.args.save_dir
        if not save_path.exists():
            save_path.mkdir()

        label_map = label_map.fillna("N/A", inplace=False)
        oldtype2newtype = {}
        oldtype2newsubtype = {}
        for _, old_type, new_type, new_subtype in label_map.itertuples(index=False):
            oldtype2newtype[old_type] = new_type
            oldtype2newsubtype[old_type] = new_subtype
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        df = pd.DataFrame({
            "index": self.test_cell_id_dict[num],
            "original label": self.test_label_dict[num],
            "cell type": [oldtype2newtype.get(p, p) for p in pred],
            "cell subtype": [oldtype2newsubtype.get(p, p) for p in pred]
        })
        df.to_csv(save_path / ("SVM_" + self.args.species + f"_{self.args.tissue}_{num}.csv"), index=False)
        print(f"output has been stored in {self.args.species}_{self.args.tissue}_{num}.csv")
