from typing import Dict
import random
import string
from itertools import combinations
from statistics import median

import numpy as np
import pandas as pd
import xarray as xr

from dtaidistance import dtw

from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.core.base import BaseTransformer


class MotifDiscovery(BaseTransformer):
    """
    This module finds motifs in several subsequences.

    :param name: Name of the motif discovery
    :type name: str
    :return: a dict with the resulting motifs and additional data
    :rtype: dict {list of xArrays, dataframe,
    """

    def __init__(self, name: str = "MotifDiscovery"):
        super().__init__(name)

    def get_params(self) -> Dict[str, object]:
        """
        Returns a list of parameters used for the motif discovery

        :return: Parameters set for the motif discovery
        :rtype: Dict
        """
        return {}

    def set_params(self):
        """
        Sets the parameters for the motif discovery

        """

    def _get_ecdf(self, x: xr.DataArray):
        """
        Method to calculate the empirical cumulative distribution function of a time series
        :param x: a numeric vector representing the univariate time series
        :type x: dataframe
        :return: ecdf function for the time series
        """
        # Drop all values = 0
        data = x[np.array(x, dtype=np.int64) != 0]
        ecdf = self._calculate_ecdf(data)
        return ecdf

    @staticmethod
    def _calculate_ecdf(data: xr.DataArray):
        """
        Method to calculate the empirical cumulative distribution function of a time series.
        Warning: This method is equal to stats::ecdf in R. The ecdf function in
        statsmodels.distributions.empirical_distribution.ECDF does not calculate the same ecdf like stats::ecdf does.

        :param data: numeric vector representing the univariate time series
        :return: (x,y) ecdf for the time series
        """
        x = np.sort(data)
        n = x.size
        y = np.arange(1, n + 1) / n
        return x, y

    @staticmethod
    def _create_eSAX(x: xr.DataArray, b: np.ndarray, w: int):
        """
        This function creates eSAX symbols for an univariate time series.

        :param x: numeric vector representing the univariate time series
        :param b: breakpoints used for the eSAX representation
        :param w: defines the word size used for the eSAX transformation
        :return: the function returns an eSAX representation of x
        """
        # Perform the piecewise aggregation
        indices = ((np.linspace(start=1, stop=len(x), num=w + 1)).round(0).astype(int)) - 1
        pieces = np.empty(shape=len(indices) - 1)

        for i in range(0, (len(indices) - 1)):
            if indices[i] == 0:
                pieces[i] = x[indices[i]]
            elif i == len(indices) - 2:
                pieces[i] = (x[indices[i]] + x[indices[i + 1]]) / 2
            else:
                pieces[i] = np.nanmean([x[indices[i]], x[indices[i + 1] - 1]])

        # Create an alphabet with double and tripple letter combinations (a, aa, aaa)
        letters = list(string.ascii_lowercase)
        alphabet = letters + [x + x for x in letters] + [x + x + x for x in letters]

        # Assign the alphabet
        let = alphabet[0:len(b)]

        # Add symbols to sequence according to breakpoints
        sym = []
        for i in range(0, len(pieces)):
            obs = pieces[i]
            temp = []
            for idx, val in enumerate(b):
                if val <= obs:
                    temp.append(idx)
            if len(temp) != 0:
                sym.append(let[max(temp)])

        return [sym, pieces, indices, b]

    def _eSAX_time_series(self, ts_subs: list, w: int, per: np.ndarray):
        """
        This method creates the eSAX representation for each subsequence
        and puts them rowwise into a dataframe.
        :param ts_subs: ts_subs a list of np arrays with the subsequences of the time-series
        :type ts_subs: list of np arrays
        :param w: defines the word size used for the eSAX transformation
        :type w: int
        :param per: percentiles depending on the ecdf of the time-series
        :type per: np.quantile
        :return: dataframe with the symbolic representations of the subsequences (rowwise)
                and the non-symbolic subsequences in pieces_all
        :rtype: pd.Dataframe and list of ndarrays
        """
        ## create eSAX time Series
        print("Creating the eSAX pieces")
        # Create list to access the SAX pieces later.
        pieces_all = []

        # initialize empty vector for the results
        ts_sax = []

        # Transformation of every subsequence in ts.subs into a symbolic aggregation.
        startpoints = [0]

        # Store the startpoint of each sequence in the original time series,
        # the startpoint is the sum of the length of all previous sequences + 1
        # for the first sequence there are no previous sequences, thus start = 1.

        for i in range(0, len(ts_subs) - 1):
            sax_temp = self._create_eSAX(x=ts_subs[i], w=w, b=per)
            startpoints.append(startpoints[i] + len(ts_subs[i]))

            # store the sax pieces
            pieces = sax_temp[1]
            pieces_all.append(pieces)
            ts_sax.append(self._create_eSAX(x=ts_subs[i], w=w, b=per)[0])

        ts_sax.append(self._create_eSAX(x=ts_subs[len(ts_subs) - 1], w=w, b=per)[0])

        ts_sax_df1 = pd.DataFrame(startpoints)
        ts_sax_df1.rename(columns={0: "StartP"}, inplace=True)
        ts_sax_df2 = pd.DataFrame(ts_sax)
        ts_sax_df = pd.concat([ts_sax_df1, ts_sax_df2], axis=1)

        print("Searching for Motifs")

        return ts_sax_df, pieces_all

    @staticmethod
    def _random_projection(ts_sax_df: pd.DataFrame, num_iterations: int):
        """
        In this method the random projection is carried out.
        Random columns of ts_sax_df are chosen (pairwise) and a collision matrix is generated

        :param ts_sax_df: dataframe with the symbolic representation of the subsequences (rowwise)
        :param num_iterations: number of iterations for the random projection (the higher that number is, the
        approximate result gets closer to the "true" result
        :return: the collision matrix
        """
        # Perform the random projection
        col_mat = np.zeros((ts_sax_df.shape[0], ts_sax_df.shape[0]))
        col_mat = pd.DataFrame(col_mat).astype(int)
        for i in range(0, num_iterations):
            random.seed(i + 42)
            col_pos = sorted(random.sample(list(ts_sax_df.columns.values)[1:], 2))
            sax_mask = pd.DataFrame(ts_sax_df.iloc[:, col_pos])
            unique_lab = sax_mask.drop_duplicates()

            mat = []
            for j in range(0, len(unique_lab.index)):
                indices = []
                for k in range(0, len(sax_mask.index) - 1):
                    indices.append(sax_mask.iloc[k, :].equals(unique_lab.iloc[j, :]))
                mat.append(indices)

            mat = pd.DataFrame(mat)

            if len(mat) != 0:
                for k in range(0, len(mat) - 1):
                    true_idx = np.where(mat.iloc[k,])
                    if len(true_idx[0]) > 1:
                        com = [n for n in combinations(true_idx[0], 2)]
                        for m in com:
                            col_mat.iloc[m[0], m[1]] += 1

        return col_mat

    @staticmethod
    def _extract_motif_pair(ts_sax_df: pd.DataFrame, col_mat: pd.DataFrame, ts_subs: list, num_iterations: int,
                           count_ratio_1: float = 5, count_ratio_2: float = 1.5,
                           max_dist_ratio: float = 2.5):
        """
        Here, the motif pairs with the highest number of collisions in the collision matrix are extracted.
        :param ts_sax_df: dataframe with the symbolic representation of the subsequences (rowwise)
        :param col_mat: collision matrix
        :param ts_subs: subsequences in a list of ndarrays
        :param num_iterations: number of iterations for the random projection
        :param count_ratio_1: first count ratio
        :param count_ratio_2: second count ratio
        :param max_dist_ratio: maximum distance ratio for determining if the euclidean distance between
                    two motif candidates is smaller than a threshold
        :return: a list of ndarrays with the starting indices of the motifs in the original time-series
        """
        # Extract the tentative motif pair
        counts = np.array([], dtype=np.int64)
        for i in range(0, col_mat.shape[1]):
            temp = col_mat.iloc[:, i]
            counts = np.concatenate((counts, temp), axis=None)
        counts = -np.sort(-counts)
        counts_sel = np.where(counts >= (num_iterations / count_ratio_1))[0]
        counts_sel = [counts[sel] for sel in counts_sel]
        counts_sel_no_dupl = sorted(set(counts_sel), reverse=True)

        motif_pair = []
        for value in counts_sel_no_dupl:
            temp = np.where(col_mat == value)
            for x, y in zip(temp[0], temp[1]):
                motif_pair.append([x, y])

        motif_pair = pd.DataFrame(motif_pair)
        if motif_pair.shape == (0, 0):
            print("No motif candidates")
            return False
        counter = 0

        indices = []
        for x, y in zip(motif_pair.iloc[:, 0], motif_pair.iloc[:, 1]):

            pair = np.array([ts_sax_df.iloc[x, 0], ts_sax_df.iloc[y, 0]])
            cand_1 = np.array(ts_subs[x])
            cand_2 = np.array(ts_subs[y])

            # Dynamic Time Warping is used for candidates of different length
            dist_raw = dtw.distance(cand_1, cand_2)

            col_no = col_mat.iloc[x, :]
            ind_cand = np.where(col_no > (max(col_no) / count_ratio_2))[0]
            ind_final = None

            if len(ind_cand) > 1:
                ind_temp = np.delete(ind_cand, np.where(ind_cand == motif_pair.iloc[counter, 1])[0])
                counter += 1
                if len(ind_temp) == 1:
                    ind_final = np.array([ts_sax_df.iloc[ind_temp[0], 0]])
                elif len(ind_temp) > 1:
                    cand_sel = []
                    dist_res = []
                    for j in ind_temp:
                        dist_res.append(dtw.distance(cand_1, ts_subs[j]))
                        cand_sel.append(ts_subs[j])
                    ind_final = ts_sax_df.iloc[
                        ind_temp[[i for i, v in enumerate(dist_res) if v <= max_dist_ratio * dist_raw]], 0].to_numpy()
            else:
                pass

            if ind_final is not None:
                pair = np.concatenate((pair, ind_final), axis=0)
                pair = np.unique(pair, axis=0)
            ind_final = None
            indices.append(pair)

        # Combine the indices if there is any overlap
        vec_subset = np.repeat(0, len(indices))
        for i in range(0, len(indices) - 1):
            for j in range(i + 1, len(indices)):
                if len(np.intersect1d(indices[i], indices[j])) > 0:
                    indices[j] = np.unique(np.concatenate((indices[i], indices[j])))
                    vec_subset[i] = 1

        indices = [indices[u] for u in np.where(vec_subset == 0)[0]]

        return indices

    def transform(self, data: xr.DataArray, ts_subs: dict):
        """
        This method combines all previous steps to extract the motifs.
        :param data:
        :param ts_subs: subsequences
        :param ecdf: euclidean cumulative distribution function
        :return: dict with subsequences, sax dataframe, motifs (symbolic, non-symbolic), collision matrix,
        indices, non-symbolic subsequences
        """
        print("Looking at 15 min data aggregation")

        # calculate the ecdf for the alphabet
        ecdf = self._get_ecdf(data)
        ecdf_df = pd.DataFrame()
        ecdf_df["x"] = ecdf[0]
        ecdf_df["y"] = ecdf[1]

        # plots.ecdf_plot(ecdf)

        ## set parameters for the eSAX algorithm
        # NOTE: According to Nicole Ludwig those parameters were set based on experience and turned
        # out to be the best working ones across 2-3 data sets
        # (e.g. count ratios have high influence but she found a good trade-off)
        # The parameters can be adapted for optimizing the algorithms quality

        breaks = 10  # number of breakpoints for the eSAX algorithm
        lengths = [len(i) for i in ts_subs]
        w = round(median(lengths) + 0.5)  # word size

        # set parameters for the random projection

        # Calculate the breakpoints for the eSAX algorithm
        # set the number of breakpoints (percentiles)
        qq = np.linspace(start=0, stop=1, num=breaks + 1)

        # store the percentiles
        per = np.quantile(ecdf_df["x"], qq)

        # use only unique percentiles for the alphabet distribution
        per = np.unique(per)

        # add the minimum as the lowest letter
        minimum = min([i.min() for i in ts_subs])
        per[0] = minimum

        # set parameters for the random projection and motif candidates
        max_length = (max(lengths) * 0.1).__round__()
        num_iterations = min(max_length, round(w / 10))

        # Create eSAX time Series
        ts_sax_df, pieces_all = self._eSAX_time_series(ts_subs, w, per)

        # Perform the random projection
        col_mat = self._random_projection(ts_sax_df, num_iterations)

        # Extract motif candidates
        indices = self._extract_motif_pair(ts_sax_df, col_mat, ts_subs, num_iterations)

        motif_raw = []
        motif_sax = []
        for val in indices:
            motif_raw_indices = np.where(np.isin(ts_sax_df.iloc[:, 0].to_numpy(), val))[0]
            motif_raw.append([ts_subs[v] for v in motif_raw_indices])
            motif_sax.append(ts_sax_df.iloc[motif_raw_indices, :])

        found_motifs = {'ts_subs': ts_subs, 'ts_sax_df': ts_sax_df, 'motif_raw': motif_raw,
                        'motif_sax': motif_sax, 'col_mat': col_mat, 'indices': indices, 'pieces_all': pieces_all}

        print("Done")

        return found_motifs