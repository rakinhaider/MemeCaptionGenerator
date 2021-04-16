# Collected from Homework skeleton.

# All imports.
import abc
import torch
import numpy as onp
import re
import math
from typing import Any
from typing import ClassVar
from typing import Tuple, List, Dict
from sklearn.model_selection import StratifiedKFold
from meme_cap_generator import MemeDataset

# =============================================================================
# *****************************************************************************
# -----------------------------------------------------------------------------
# ## Data Structures
# -----------------------------------------------------------------------------
# *****************************************************************************
# =============================================================================


class KFoldStructure(
    torch.utils.data.Dataset,
    metaclass=type,
):
    r"""
    K-fold data structure.
    """
    # =========================================================================
    # -------------------------------------------------------------------------
    # =========================================================================
    # /
    # ANNOTATE
    # /
    TRAIN: ClassVar[int]
    VALIDATE: ClassVar[int]
    TEST: ClassVar[int]

    # Train, validation, test split constants.
    TRAIN = 0
    VALIDATE = 1
    TEST = 2

    def __init__(
        self,
        /,
        memory: MemeDataset, split_usage: int,
        *,
        rest_test_split: int, train_valid_split: int,
        rest_test_index: int, train_valid_index: int,
    ) -> None:
        r"""
        Initialize the class.

        Args
        ----
        - memory :
            In-memory data.
        - split_usage :
            K-fold split usage.
            It should be an integer representing training, validation or test.
        - rest_test_split :
            K-fold split proportion of rest over test.
        - train_valid_split :
            K-fold split proportion of training over validation.
        - rest_test_index :
            K-fold split index of rest and test split.
        - train_valid_index :
            K-fold split index of training and validation.

        Returns
        -------
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        self.memory: MemeDataset
        self.rest_test_split: int
        self.train_valid_split: int
        self.rest_test_index: int
        self.train_valid_index: int
        self.split_usage: int
        self.indices: List[int]
        self.distributions: List[float]

        # Save necessary attributes.
        self.memory = memory
        self.rest_test_split = rest_test_split
        self.train_valid_split = train_valid_split
        self.rest_test_index = rest_test_index
        self.train_valid_index = train_valid_index
        self.split_usage = split_usage

        # Get memory virtual split table.
        self.indices, self.distributions = self.get_virtual_split_table()

    def __len__(
        self,
        /,
    ) -> int:
        r"""
        Get length.

        Args
        ----

        Returns
        -------
        - length :
            Length.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # Get length directly.
        return len(self.indices)

    def __repr__(
        self,
        /,
    ) -> str:
        r"""
        Get representation string.

        Args
        ----

        Returns
        -------
        - msg :
            Representation string.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # Get representation string directly by length and label distribution.
        return "[{:s}] \033[91m{:d}\033[0m".format(
            ", ".join(["{:.3f}".format(itr) for itr in self.distributions]),
            len(self.indices),
        )

    @classmethod
    def decolor(
        cls,
        /,
        msg: str,
    ) -> str:
        r"""
        Remove color from string.

        Args
        ----
        - msg :
            Message string.

        Returns
        -------
        - msg :
            Decolorized message string.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # Remove color style string.
        return re.sub(r"\033\[[^m]+m", "", msg)

    # =========================================================================
    # -------------------------------------------------------------------------
    # Generate virtual split address mapping.
    # -------------------------------------------------------------------------
    # =========================================================================

    def get_virtual_split_table(
        self,
        /,
    ) -> onp.ndarray:
        r"""
        Get the virtual split table matching split index to memory index.

        Args
        ----

        Returns
        -------
        - virtual_split
            Virtual split table matching split index to memory index.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        label_array: onp.ndarray
        label_bin: onp.ndarray
        label_min: int
        label_max: int
        label_num: int
        # -----
        total_indices: onp.ndarray
        rest_indices: onp.ndarray
        test_indices: onp.ndarray
        train_indices: onp.ndarray
        valid_indices: onp.ndarray
        # -----
        indices: onp.ndarray
        distributions: onp.ndarray

        # Treat all targets as labels.
        label_array = self.get_label_array()
        label_bin = onp.unique(label_array)
        label_num = len(label_bin)
        label_min = label_bin.min().item()
        label_max = label_bin.max().item()
        if (label_min == 0 and label_max == label_num - 1):
            pass
        else:
            print(
                "[\033[91mError\033[0m]: K-fold split requires that labels" \
                " are consecutive from 0."
            )
            raise RuntimeError

        # Set all indices to be considered.
        total_indices = onp.arange(len(label_array)).astype(int)

        # Get rest and test split.
        if (self.memory.test_indices is None):
            rest_indices, test_indices = self.split(
                total_indices, label_array,
                self.rest_test_split, self.rest_test_index,
            )
        else:
            test_indices = onp.array(self.memory.test_indices).astype(int)
            rest_indices = onp.delete(total_indices, self.memory.test_indices)

        # Get training and validation split.
        if (self.memory.valid_indices is None):
            train_indices, valid_indices = self.split(
                total_indices[rest_indices], label_array[rest_indices],
                self.train_valid_split, self.train_valid_index,
            )
        else:
            valid_indices = onp.array(self.memory.valid_indices).astype(int)
            train_indices = onp.delete(
                total_indices, onp.union1d(valid_indices, test_indices),
            )

        # Ensure no intersection.
        if (len(onp.intersect1d(valid_indices, test_indices, True)) > 0):
            print(
                "[\033[91mError\033[0m]: Encounter intersection between" \
                " validation and test."
            )
            raise RuntimeError
        else:
            pass

        # Get indices and label distributions of focusing usage.
        indices = (
            train_indices, valid_indices, test_indices,
        )[self.split_usage]
        distributions = onp.zeros(label_num)
        onp.add.at(distributions, label_array[indices], 1)
        distributions = distributions / distributions.sum()
        return indices.tolist(), distributions.tolist()

    @abc.abstractmethod
    def get_label_array(
        self,
        /,
    ) -> onp.ndarray:
        r"""
        Get all lables to be split as an array.

        Args
        ----

        Returns
        -------
        - label_array :
            Label array.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # VIRTUAL
        # /
        ...

    def split(
        self,
        indices: onp.ndarray, labels: onp.ndarray, prop: int, index: int,
        /,
    ) -> Tuple[onp.ndarray, onp.ndarray]:
        r"""
        Split given indices based on balanced label allocation.

        Args
        ----
        - indices :
            Indices to be split into majority and minority parts.
        - labels :
            Labels correspond to the indices.
        - prop :
            Proportion of majority over minority.
        - index :
            Index of the split.

        Returns
        -------
        - major_indices :
            Indices of majority part.
        - minor_indices :
            Indices of minority part.

        It will take the ceil in the minority part can not properly divide
        given indices.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        op: StratifiedKFold
        major_indices: onp.ndarray
        minor_indices: onp.ndarray

        # Get split operator.
        op = StratifiedKFold(n_splits=prop, shuffle=False)

        # Get the correct split.
        for _, (major_indices, minor_indices) in zip(
            range(index + 1), iter(op.split(indices, labels)),
        ):
            pass
        return indices[major_indices], indices[minor_indices]

    def preprocess(
        self,
        kwargs: Dict[str, Any],
        /,
    ) -> None:
        r"""
        Preprocess data in the split.

        Args
        ----
        - kwargs :
            Keyword arguments for preprocessing.

        Returns
        -------
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # By default, there is no preprocessing.
        pass


class CoraDataStructure(
    KFoldStructure,
    metaclass=type,
):
    r"""
    Cora data structure.
    """
    # =========================================================================
    # -------------------------------------------------------------------------
    # =========================================================================
    def __getitem__(
        self,
        /,
        i: int,
    ) -> int:
        r"""
        Index an element.

        Args
        ----
        - i :
            Element index.

        Returns
        -------
        - element :
            Element.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # We only need node index as a single sample.
        return self.indices[i]

    def get_label_array(
        self,
        /,
    ) -> onp.ndarray:
        r"""
        Get all lables to be split as an array.

        Args
        ----

        Returns
        -------
        - label_array :
            Label array.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        label_tensor: torch.Tensor

        # Get the label tensor.
        _, _, label_tensor = next(iter(self.memory))
        return label_tensor.numpy()

    @staticmethod
    def collate(
        samples: List[int],
    ) -> Dict[str, torch.Tensor]:
        r"""
        Collation function.

        Args
        ----
        - samples :
            Samples to be collated as a batch.

        Returns
        -------
        - batch
            Batch.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        indices: torch.Tensor

        # Samples are just available node indices.
        return dict(indices=torch.LongTensor(samples))


class PTBDataStructure(
    KFoldStructure,
    metaclass=type,
):
    r"""
    PTB data structure.
    """
    # =========================================================================
    # -------------------------------------------------------------------------
    # =========================================================================
    def __len__(
        self,
        /,
    ) -> int:
        r"""
        Get length.

        Args
        ----

        Returns
        -------
        - length :
            Length.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # Length is number of truncated BPTT chunks.
        return self.num_chunks

    def __repr__(
        self,
        /,
    ) -> str:
        r"""
        Get representation string.

        Args
        ----

        Returns
        -------
        - msg :
            Representation string.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # Get representation string directly by length and label distribution.
        if (self.truncate == 0):
            return "\033[91m{:d}\033[0m".format(self.length)
        else:
            return "\033[91m{:d}\033[0m[{:d}x{:d}, ...]".format(
                self.num_chunks, self.truncate, self.batch_size,
            )

    def __getitem__(
        self,
        /,
        i: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Index an element.

        Args
        ----
        - i :
            Element index.

        Returns
        -------
        - element :
            Element.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        inf: int
        sup: int
        input: torch.Tensor
        target: torch.Tensor

        # Detect Markov and LSTM cases.
        if (self.truncate == 0):
            # Use the only chunk.
            return self.input, self.target
        else:
            # Locate the chunk boundaries.
            inf = self.truncate * i
            sup = min(inf + self.truncate, self.length // self.batch_size)
            input = self.input[inf:sup]
            target = self.target[inf:sup]
            return input, target

    def get_label_array(
        self,
        /,
    ) -> onp.ndarray:
        r"""
        Get all lables to be split as an array.

        Args
        ----

        Returns
        -------
        - label_array :
            Label array.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # Get null split labels directly.
        return onp.zeros(len(self.memory)).astype(int)

    def preprocess(
        self,
        kwargs: Dict[str, Any],
        homework: Any,
        /,
    ) -> None:
        r"""
        Preprocess data in the split.

        Args
        ----
        - kwargs :
            Keyword arguments for preprocessing.

        Returns
        -------
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        self.batch_size: int
        self.truncate: int
        index: int
        self.input: torch.Tensor
        self.target: torch.Tensor
        self.length: int
        self.num_chunks: int

        # Get batch size and truncation length.
        self.batch_size = kwargs["batch_size"]
        self.truncate = kwargs["truncate"]

        # Locate raw data and move it to local structure.
        index, = self.indices
        self.length = len(self.memory.memory[index])

        # Detect Markov and LSTM cases.
        if (self.truncate == 0):
            # Markov requires specific batch settings.
            if (self.batch_size == 1):
                pass
            else:
                print(
                    "[\033[91mError\033[0m]: Markovian model requires batch" \
                    " size being 1.",
                )
                raise RuntimeError
            self.input = self.memory.memory[index].view(self.length)
            self.target = self.memory.memory[index].view(self.length)
            self.num_chunks = 1
        else:
            # Split input and target by one-step gap.
            index, = self.indices
            self.length = len(self.memory.memory[index])
            self.input = self.memory.memory[index][:-1]
            self.target = self.memory.memory[index][1:]

            # Cut the whole sequence into several batches (sub sequences).
            self.length, self.input, self.target, self.num_chunks = homework(
                self.length, self.input, self.target,
                self.batch_size, self.truncate,
            )
        # Remove raw data.
        self.memory.memory[index] = torch.Tensor([0])

    @staticmethod
    def collate(
        samples: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        r"""
        Collation function.

        Args
        ----
        - samples :
            Samples to be collated as a batch.

        Returns
        -------
        - batch
            Batch.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        input: torch.Tensor
        target: torch.Tensor

        # BPTT batch size must be 1.
        (input, target), = samples
        return dict(input=input, target=target)
