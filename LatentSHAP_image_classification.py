from scipy.sparse import issparse
import shap
from shap.utils._legacy import convert_to_instance, convert_to_model, match_instance_to_data, match_model_to_data
from shap.utils._legacy import convert_to_instance_with_index, convert_to_link, IdentityLink, convert_to_data, DenseData, \
    SparseData
from shap.utils import safe_isinstance
from scipy.special import binom
import numpy as np
import pandas as pd
import scipy as sp
import logging
import copy
import itertools
import warnings
from sklearn.linear_model import LinearRegression
import sklearn
from tqdm.auto import tqdm
import gc



log = logging.getLogger('shap')

class LatentExplainerImgs(shap.KernelExplainer):
    """
    Utilize the Kernel SHAP method to explain the output of any function.
    Kernel SHAP is a method that uses a special weighted linear regression
    to compute the importance of each feature.
    The computed importance values are Shapley values from game theory and also coefficents from a local linear
    regression.
    In order to compute the latent importance values, LatentSHAP follows the same
    steps as KernelExplainer to build a weighted linear regression for the latent space, but
    the coalitions predictions for the latent space approximated by the original space coalitions predictions.
    Parameters
    ----------
    model : predict probability function of a machine learning model.

    bg_data : numpy.array or pandas.DataFrame.
        The background dataset to use for integrating out features.
        To determine the impact of a feature, that feature is set to "missing" and the change in the model output
        is observed.
    T_func: A transformation function from the original space to the latent feature space.

    estimation_method: (String) (for now - one of ["Euclidean distance", "regression", "sklearn"])
    A similarity (or distance) method to model the statistical relationships between the non-interpretable feature space
    and the interpretable feature space through the latent background dataset.
    if estimation_method is "sklearn" - an unfitted sklearn estimator with parameters need to be passed

    proximity_model: unfitted sklearn estimator with parameters (valid only if estimation_method is "sklearn")

    link : (String) "identity" or "logit"
    A generalized linear model link to connect the feature importance values to the model
    output. Since the feature importance values, phi, sum up to the model output, it often makes
    sense to connect them to the output with a link function where link(output) = sum(phi).
    If the model output is a probability then the LogitLink link function makes the feature
    importance values have log-odds units.

    preds_bg: model's prediction for the background dataset

    kwargs: (Dictionary)
    possible keys-
    *keep_index - keep the index of the data
    *keep_index_ordered - keep the index of the data ordered
    """
    def __init__(self, model, transformation, background_data,
                 estimation_method="sklearn", proximity_model=None,link=IdentityLink(), **kwargs):
        # convert incoming inputs to standardized iml objects

        self.model, self.transformation = model, transformation
        self.keep_index = kwargs.get("keep_index", False)
        self.keep_index_ordered = kwargs.get("keep_index_ordered", False)
        self.flatten_imgs = lambda imgs: imgs.reshape(imgs.shape[0], imgs.shape[1] * imgs.shape[2])
        self.resize_back_imgs = lambda imgs: imgs.reshape(imgs.shape[0], int(np.sqrt(imgs.shape[1])), int(np.sqrt(imgs.shape[1])), 1)
        transformed_bg_data, preds_bg = transformation(background_data), model(background_data)
        self.preds_sampled_coalitions = preds_bg #preds_sampled_coalitions
        self.transformed_bg_data = convert_to_data(transformed_bg_data, keep_index=self.keep_index)
        self.link = convert_to_link(link)
        self.sklearn_model = proximity_model
        self.sklearn_model.fit(X=transformed_bg_data, y=preds_bg.flatten())
        self.estimation_method = estimation_method

        # enforce our current input type limitations
        assert isinstance(self.transformed_bg_data, DenseData) or isinstance(self.transformed_bg_data, SparseData), \
            "This explainer only supports the DenseData and SparseData input."
        assert not self.transformed_bg_data.transposed, "This explainer does not support transposed DenseData or SparseData."

        # warn users about large background data sets
        if len(self.transformed_bg_data.weights) > 100:
            log.warning("Using " + str(len(self.transformed_bg_data.weights)) + " background data samples could cause " +
                        "slower run times. Consider using shap.sample(data, K) or shap.kmeans(data, K) to " +
                        "summarize the background as K samples.")

        # init our parameters
        self.N = self.transformed_bg_data.data.shape[0]
        # self.P = self.data.data.shape[1]
        # self.P_latent = self.transformed_bg_data.data.shape[1]
        self.linkfv = np.vectorize(self.link.f)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0
        self.nsamples = kwargs.get("nsamples", "auto")
        self.nsamples_latent = kwargs.get("nsamples_latent", "auto")

        # find E_x[f(x)]
        if isinstance(preds_bg, (pd.DataFrame, pd.Series)):
            preds_bg = np.squeeze(preds_bg.values)
        if safe_isinstance(preds_bg, "tensorflow.python.framework.ops.EagerTensor"):
            preds_bg = preds_bg.numpy()
        self.fnull = np.sum((preds_bg.T * self.transformed_bg_data.weights).T, 0)
        self.expected_value = self.linkfv(self.fnull)

        # see if we have a vector output
        self.vector_out = True
        if len(self.fnull.shape) == 0:
            self.vector_out = False
            self.fnull = np.array([self.fnull])
            self.D = 1
            self.expected_value = float(self.expected_value)
        else:
            self.D = self.fnull.shape[0]


    def latent_shap_values(self, X, **kwargs):
        '''
        Estimate the latent SHAP values for a set of samples.
        for each sample from X, calculate feature importance with 'explain' func
        Parameters
        ----------
        X : numpy.array or pandas.DataFrame or any scipy.sparse matrix
            A matrix of samples (# samples x # features) on which to explain the model's output.

        kwargs: (Dictionary)
        possible keys-
        *nsamples - ("auto" or int, default "auto") number of samples for the original space
        *nsamples_latent - ("auto" or int, default "auto") number of samples for the original space
        *silent - (boolean, default False) - disable tdqm progress bar if true
        *gc_collect - (boolean, default False) - Run garbage collection after each explanation round. Sometime needed for memory intensive explanations.
        *l1_reg - ("auto" or boolean, default "auto") if val = False or 0 then do not regulate the linear reg. if val = "auto", do l1 regulation if nsamples < 0.2*max samples. else - do regularition
        '''

        pred_for_instances, transformed_instances = self.model(X), self.transformation(X)
        X = self.flatten_imgs(X)
        # explain the whole dataset
        explanations_latent = []
        for i in tqdm(range(transformed_instances.shape[0]), disable=kwargs.get("silent", False)):
            pred_for_instance = pred_for_instances[i:i+1]
            transformed_instance = transformed_instances[i:i+1,:]
            original_space_instance = X[i:i+1, :] if X is not None else None
            abstract_explanation = self.explain(transformed_instance, pred_for_instance, original_space_instance, **kwargs)
            explanations_latent.append(abstract_explanation)
            if kwargs.get("gc_collect", False):
                gc.collect()

        # vector-output
        s_latent = explanations_latent[0].shape
        outs_latent = [np.zeros((pred_for_instances.shape[0], s_latent[0])) for j in range(s_latent[1])]
        for i in range(pred_for_instances.shape[0]):
            for j in range(s_latent[1]):
                outs_latent[j][i] = explanations_latent[i][:, j]
        return outs_latent

    '''
    Auxilary functions for 'shap_values' func:
    '''

    def handle_varying_features(self, varyingInds, data):
        # find the feature groups we will test. If a feature does not change from its
        # current value then we know it doesn't impact the model
        if data.groups is None:
            varyingFeatureGroups = np.array([i for i in varyingInds])
            M = varyingFeatureGroups.shape[0]
        else:
            varyingFeatureGroups = [data.groups[i] for i in varyingInds]
            M = len(varyingFeatureGroups)
            groups = data.groups
            # convert to numpy array as it is much faster if not jagged array (all groups of same length)
            if varyingFeatureGroups and all(len(groups[i]) == len(groups[0]) for i in varyingInds):
                varyingFeatureGroups = np.array(varyingFeatureGroups)
                # further performance optimization in case each group has a single value
                if varyingFeatureGroups.shape[1] == 1:
                    varyingFeatureGroups = varyingFeatureGroups.flatten()
        return varyingFeatureGroups, M

    def explain(self, transformed_instance, pred_for_instance, original_space_instance=None, **kwargs): #incoming_instance
        '''
        :param incoming_instance: an instance to explain
        :param
        kwargs: as in 'shap_values' func
            possible keys:
                pred_for_incoming_instance=None
                transformed_incoming_instance=None

        :return: feature importnce for this instance
        Include * steps:
        1. convert the instance to SHAP's instance, find the varying features
        '''

        instance_latent = transformed_instance
        instance_latent = convert_to_instance(instance_latent)
        match_instance_to_data(instance_latent, self.transformed_bg_data)
        self.varyingInds_latent = self.varying_groups_latent(instance_latent.x)
        self.varyingFeatureGroups_latent, self.M_latent = self.handle_varying_features(self.varyingInds_latent, self.transformed_bg_data)

        if isinstance(pred_for_instance, (pd.DataFrame, pd.Series)):
            pred_for_instance = pred_for_instance.values
        self.fx = pred_for_instance[0]

        if not self.vector_out:
            self.fx = np.array([self.fx])

        # if no latent features vary then no feature has an effect
        if self.M_latent == 0:
            phi = np.zeros((self.transformed_bg_data.groups_size, self.D))
            phi_var = np.zeros((self.transformed_bg_data.groups_size, self.D))

        # if only one latent feature varies then it has all the effect
        elif self.M_latent == 1:
            phi = np.zeros((self.transformed_bg_data.groups_size, self.D))
            phi_var = np.zeros((self.transformed_bg_data.groups_size, self.D))
            diff = self.link.f(self.fx) - self.link.f(self.fnull)
            for d in range(self.D):
                phi[self.varyingInds_latent[0], d] = diff[d]

        # if more than one feature varies then we have to do real work
        else:
            self.l1_reg = kwargs.get("l1_reg", "auto")


            def get_number_of_samples(M, nsamples):
                # pick a reasonable number of samples if the user didn't specify how many they wanted
                if nsamples == "auto":
                    nsamples = 2 * M + 2 ** 15

                # if we have enough samples to enumerate all subsets then ignore the unneeded samples
                max_samples = 2 ** 30
                if M <= 30:
                    max_samples = 2 ** M - 2
                if nsamples > max_samples: nsamples = max_samples
                return nsamples, max_samples

            self.nsamples_latent, self.max_samples_latent = get_number_of_samples(self.M_latent, self.nsamples_latent)

            # reserve space for some of our computations
            self.allocate()

            #If we generate samples for original space - execute self.add_local_samples with latent=False

            # add local samples and thier weights - for latent space:
            self.add_local_samples(instance_latent, latent=True)
            # execute the model on the synthetic samples we have created
            self.run()

            # solve then expand the feature importance (Shapley value) vector to contain the non-varying features
            phi_latent = np.zeros((self.transformed_bg_data.groups_size, self.D))
            phi_var_latent = np.zeros((self.transformed_bg_data.groups_size, self.D))
            for d in range(self.D):
                vphi_latent, vphi_var_latent = self.solve_latent(1, d)
                phi_latent[self.varyingInds_latent, d] = vphi_latent
                phi_var_latent[self.varyingInds_latent, d] = vphi_var_latent

        return phi_latent

    def varying_groups_latent(self, x):
        if not sp.sparse.issparse(x):
            varying = np.zeros(self.transformed_bg_data.groups_size)
            for i in range(0, self.transformed_bg_data.groups_size):
                inds = self.transformed_bg_data.groups[i]
                x_group = x[0, inds]
                if sp.sparse.issparse(x_group):
                    if all(j not in x.nonzero()[1] for j in inds):
                        varying[i] = False
                        continue
                    x_group = x_group.todense()
                num_mismatches = np.sum(np.frompyfunc(self.not_equal, 2, 1)(x_group, self.transformed_bg_data.data[:, inds]))
                varying[i] = num_mismatches > 0
            varying_indices = np.nonzero(varying)[0]
            return varying_indices
        else:
            varying_indices = []
            # go over all nonzero columns in background and evaluation data
            # if both background and evaluation are zero, the column does not vary
            varying_indices = np.unique(np.union1d(self.transformed_bg_data.data.nonzero()[1], x.nonzero()[1]))
            remove_unvarying_indices = []
            for i in range(0, len(varying_indices)):
                varying_index = varying_indices[i]
                # now verify the nonzero values do vary
                data_rows = self.transformed_bg_data.data[:, [varying_index]]
                nonzero_rows = data_rows.nonzero()[0]

                if nonzero_rows.size > 0:
                    background_data_rows = data_rows[nonzero_rows]
                    if sp.sparse.issparse(background_data_rows):
                        background_data_rows = background_data_rows.toarray()
                    num_mismatches = np.sum(np.abs(background_data_rows - x[0, varying_index]) > 1e-7)
                    # Note: If feature column non-zero but some background zero, can't remove index
                    if num_mismatches == 0 and not \
                            (np.abs(x[0, [varying_index]][0, 0]) > 1e-7 and len(nonzero_rows) < data_rows.shape[0]):
                        remove_unvarying_indices.append(i)
            mask = np.ones(len(varying_indices), dtype=bool)
            mask[remove_unvarying_indices] = False
            varying_indices = varying_indices[mask]
            return varying_indices

    def allocate(self):

        self.synth_latent_data = np.tile(self.transformed_bg_data.data, (self.nsamples_latent, 1))
        self.maskMatrix_latent = np.zeros((self.nsamples_latent, self.M_latent))
        self.kernelWeights_latent = np.zeros(self.nsamples_latent)
        self.nsamplesAdded_latent = 0
        self.nsamplesRun_latent = 0

        self.y = self.preds_sampled_coalitions
        if isinstance(self.y, (pd.DataFrame, pd.Series)):
            self.y = self.y.values


    def add_local_samples(self, instance, latent):
        if latent==False:
            M = self.M
            nsamples = self.nsamples
            addsample = self.addsample
            nsamplesAdded = self.nsamplesAdded
            kernelWeights = self.kernelWeights
        else:
            M = self.M_latent
            nsamples = self.nsamples_latent
            addsample = self.addsample_latent
            nsamplesAdded = self.nsamplesAdded_latent
            kernelWeights = self.kernelWeights_latent



        # weight the different subset sizes
        num_subset_sizes = np.int(np.ceil((M - 1) / 2.0))
        num_paired_subset_sizes = np.int(np.floor((M - 1) / 2.0))
        weight_vector = np.array([(M - 1.0) / (i * (M - i)) for i in range(1, num_subset_sizes + 1)])
        weight_vector[:num_paired_subset_sizes] *= 2
        weight_vector /= np.sum(weight_vector)
        log.debug("weight_vector = {0}".format(weight_vector))
        log.debug("num_subset_sizes = {0}".format(num_subset_sizes))
        log.debug("num_paired_subset_sizes = {0}".format(num_paired_subset_sizes))
        log.debug("M = {0}".format(M))

        # fill out all the subset sizes we can completely enumerate
        # given nsamples*remaining_weight_vector[subset_size]
        num_full_subsets = 0
        num_samples_left = nsamples
        group_inds = np.arange(M, dtype='int64')
        mask = np.zeros(M)
        remaining_weight_vector = copy.copy(weight_vector)

        for subset_size in range(1, num_subset_sizes + 1):

            # determine how many subsets (and their complements) are of the current size
            nsubsets = binom(M, subset_size)
            if subset_size <= num_paired_subset_sizes: nsubsets *= 2
            log.debug("subset_size = {0}".format(subset_size))
            log.debug("nsubsets = {0}".format(nsubsets))
            log.debug("self.nsamples*weight_vector[subset_size-1] = {0}".format(
                num_samples_left * remaining_weight_vector[subset_size - 1]))
            log.debug("self.nsamples*weight_vector[subset_size-1]/nsubsets = {0}".format(
                num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets))

            # see if we have enough samples to enumerate all subsets of this size
            if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
                num_full_subsets += 1
                num_samples_left -= nsubsets

                # rescale what's left of the remaining weight vector to sum to 1
                if remaining_weight_vector[subset_size - 1] < 1.0:
                    remaining_weight_vector /= (1 - remaining_weight_vector[subset_size - 1])

                # add all the samples of the current subset size
                w = weight_vector[subset_size - 1] / binom(M, subset_size)
                if subset_size <= num_paired_subset_sizes: w /= 2.0
                for inds in itertools.combinations(group_inds, subset_size):
                    mask[:] = 0.0
                    mask[np.array(inds, dtype='int64')] = 1.0
                    addsample(instance.x, mask, w)
                    if latent == False:
                        nsamplesAdded = self.nsamplesAdded
                        kernelWeights = self.kernelWeights
                    else:
                        nsamplesAdded = self.nsamplesAdded_latent
                        kernelWeights = self.kernelWeights_latent

                    if subset_size <= num_paired_subset_sizes:
                        mask[:] = np.abs(mask - 1)
                        addsample(instance.x, mask, w)
                        if latent == False:
                            nsamplesAdded = self.nsamplesAdded
                            kernelWeights = self.kernelWeights
                        else:
                            nsamplesAdded = self.nsamplesAdded_latent
                            kernelWeights = self.kernelWeights_latent
            else:
                break
        log.info("num_full_subsets = {0}".format(num_full_subsets))

        # add random samples from what is left of the subset space
        nfixed_samples = nsamplesAdded
        samples_left = nsamples - nsamplesAdded
        log.debug("samples_left = {0}".format(samples_left))

        if num_full_subsets != num_subset_sizes:
            remaining_weight_vector = copy.copy(weight_vector)
            remaining_weight_vector[:num_paired_subset_sizes] /= 2  # because we draw two samples each below
            remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
            remaining_weight_vector /= np.sum(remaining_weight_vector)
            log.info("remaining_weight_vector = {0}".format(remaining_weight_vector))
            log.info("num_paired_subset_sizes = {0}".format(num_paired_subset_sizes))
            ind_set = np.random.choice(len(remaining_weight_vector), 4 * samples_left,
                                       p=remaining_weight_vector)
            ind_set_pos = 0
            used_masks = {}
            while samples_left > 0 and ind_set_pos < len(ind_set):
                mask.fill(0.0)
                ind = ind_set[
                    ind_set_pos]  # we call np.random.choice once to save time and then just read it here
                ind_set_pos += 1
                subset_size = ind + num_full_subsets + 1
                mask[np.random.permutation(M)[:subset_size]] = 1.0

                # only add the sample if we have not seen it before, otherwise just
                # increment a previous sample's weight
                mask_tuple = tuple(mask)
                new_sample = False
                if mask_tuple not in used_masks:
                    new_sample = True
                    used_masks[mask_tuple] = nsamplesAdded
                    samples_left -= 1
                    addsample(instance.x, mask, 1.0)
                else:
                    kernelWeights[used_masks[mask_tuple]] += 1.0

                if latent == False:
                    nsamplesAdded = self.nsamplesAdded
                    kernelWeights = self.kernelWeights
                else:
                    nsamplesAdded = self.nsamplesAdded_latent
                    kernelWeights = self.kernelWeights_latent

                # add the compliment sample
                if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                    mask[:] = np.abs(mask - 1)

                    # only add the sample if we have not seen it before, otherwise just
                    # increment a previous sample's weight
                    if new_sample:
                        samples_left -= 1
                        addsample(instance.x, mask, 1.0)
                    else:
                        # we know the compliment sample is the next one after the original sample, so + 1
                        kernelWeights[used_masks[mask_tuple] + 1] += 1.0

                    if latent == False:
                        nsamplesAdded = self.nsamplesAdded
                        kernelWeights = self.kernelWeights
                    else:
                        nsamplesAdded = self.nsamplesAdded_latent
                        kernelWeights = self.kernelWeights_latent

            # normalize the kernel weights for the random samples to equal the weight left after
            # the fixed enumerated samples have been already counted
            weight_left = np.sum(weight_vector[num_full_subsets:])
            log.info("weight_left = {0}".format(weight_left))
            kernelWeights[nfixed_samples:] *= weight_left / kernelWeights[nfixed_samples:].sum()

    def addsample_latent(self, x, m, w):
        offset = self.nsamplesAdded_latent * self.N
        if isinstance(self.varyingFeatureGroups_latent, (list,)):
            for j in range(self.M_latent):
                for k in self.varyingFeatureGroups_latent[j]:
                    if m[j] == 1.0:
                        self.synth_latent_data[offset:offset + self.N, k] = x[0, k]
        else:
            # for non-jagged numpy array we can significantly boost performance
            mask = m == 1.0
            groups = self.varyingFeatureGroups_latent[mask]
            if len(groups.shape) == 2:
                for group in groups:
                    self.synth_latent_data[offset:offset + self.N, group] = x[0, group]
            else:
                # further performance optimization in case each group has a single feature
                evaluation_data = x[0, groups]
                # In edge case where background is all dense but evaluation data
                # is all sparse, make evaluation data dense
                if sp.sparse.issparse(x) and not sp.sparse.issparse(self.synth_latent_data):
                    evaluation_data = evaluation_data.toarray()
                self.synth_latent_data[offset:offset + self.N, groups] = evaluation_data
        self.maskMatrix_latent[self.nsamplesAdded_latent, :] = m
        self.kernelWeights_latent[self.nsamplesAdded_latent] = w
        self.nsamplesAdded_latent += 1

    def run(self):

        self.ey_latent = np.zeros((self.nsamples_latent, self.D))

        coalition_size = self.N

        # Take the Trained ML model for each class (classes - columns in y)
        labels_amount = 1 if len(self.y.shape) == 1 else self.y.shape[1] #The amount of labels is one if y has one dimension, else the number of columns in y
        if labels_amount == 0: labels_amount = 1


        latent_labels_approximated = self.sklearn_model.predict(self.synth_latent_data)

        coalition_amount = int(len(latent_labels_approximated)/coalition_size)
        for i in range(coalition_amount): #each coalition target is the average for all the targets in the same coalition
            avg_lbl_for_coalition = np.average(latent_labels_approximated[i*coalition_size:(i+1)*coalition_size])
            if labels_amount > 1:
                self.ey_latent[i, :-1] = avg_lbl_for_coalition
            else:
                self.ey_latent[i] = avg_lbl_for_coalition
        if labels_amount > 1:
            #If there is more than one labels - calculate the nth' labels: 1 - the rest
            self.ey_latent[:, -1] = 1 - np.sum(self.ey_latent[:, :-1], axis=1)

    def solve_latent(self, fraction_evaluated, dim):
        eyAdj = self.linkfv(self.ey_latent[:, dim]) - self.link.f(self.fnull[dim])
        s = np.sum(self.maskMatrix_latent, 1)

        # do feature selection if we have not well enumerated the space
        nonzero_inds = np.arange(self.M_latent)
        log.debug("fraction_evaluated = {0}".format(fraction_evaluated))

        # eliminate one variable with the constraint that all features sum to the output
        eyAdj2 = eyAdj - self.maskMatrix_latent[:, nonzero_inds[-1]] * (
                self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim]))
        etmp = np.transpose(np.transpose(self.maskMatrix_latent[:, nonzero_inds[:-1]]) - self.maskMatrix_latent[:, nonzero_inds[-1]])
        log.debug("etmp[:4,:] {0}".format(etmp[:4, :]))

        # solve a weighted least squares equation to estimate phi
        tmp = np.transpose(np.transpose(etmp) * np.transpose(self.kernelWeights_latent))
        etmp_dot = np.dot(np.transpose(tmp), etmp)
        try:
            tmp2 = np.linalg.inv(etmp_dot)
        except np.linalg.LinAlgError:
            tmp2 = np.linalg.pinv(etmp_dot)
            warnings.warn(
                "Linear regression equation is singular, Moore-Penrose pseudoinverse is used instead of the regular inverse.\n"
                "To use regular inverse do one of the following:\n"
                "1) turn up the number of samples,\n"
                "2) turn up the L1 regularization with num_features(N) where N is less than the number of samples,\n"
                "3) group features together to reduce the number of inputs that need to be explained."
            )
        w = np.dot(tmp2, np.dot(np.transpose(tmp), eyAdj2))
        log.debug("np.sum(w) = {0}".format(np.sum(w)))
        log.debug("self.link(self.fx) - self.link(self.fnull) = {0}".format(
            self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])))
        log.debug("self.fx = {0}".format(self.fx[dim]))
        log.debug("self.link(self.fx) = {0}".format(self.link.f(self.fx[dim])))
        log.debug("self.fnull = {0}".format(self.fnull[dim]))
        log.debug("self.link(self.fnull) = {0}".format(self.link.f(self.fnull[dim])))
        phi = np.zeros(self.M_latent)
        phi[nonzero_inds[:-1]] = w
        phi[nonzero_inds[-1]] = (self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])) - sum(w)
        log.info("phi = {0}".format(phi))

        # clean up any rounding errors
        for i in range(self.M_latent):
            if np.abs(phi[i]) < 1e-10:
                phi[i] = 0

        return phi, np.ones(len(phi))