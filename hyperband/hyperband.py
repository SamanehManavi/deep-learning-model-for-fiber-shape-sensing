import numpy as np

from random import random
from math import log, ceil
from time import time, ctime
import pathlib

from data_loader import data_manager as dm

class Hyperband:

    def __init__(self, get_params_function, try_params_function):
        self.get_params = get_params_function
        self.try_params = try_params_function

        self.max_iter = 256 #81  # maximum iterations per configuration (also: how many configurations will be tried)
        self.eta = 4 # 3  # defines configuration downsampling rate (default = 3) (-> e.g. max_iter = 81, then it will go: 81, 9, 3, 1)

        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

        self.results = []  # list of dicts
        self.counter = 0
        self.best_loss = np.inf
        self.best_accuracy = 0
        self.best_counter = -1

    # can be called multiple times
    def run(self, skip_last=0, dry_run=False, fixed_params=None):

        print("Load Data...")
        data_train = dm.LoadData(args=fixed_params, csvpath=fixed_params.traincsv_path)
        data_validate = dm.LoadData(args=fixed_params, csvpath=fixed_params.validatecsv_path)
        print("************* training_path ", fixed_params.traincsv_path, ", ***********************")
        print("************* validation_path ", fixed_params.validatecsv_path, ", ***********************")

        print("Done.")

        for s in reversed(range(self.s_max + 1)):

            # initial number of configurations
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))
            print(n)

            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # n random configurations
            T = [self.get_params() for i in range(n)]


            for i in range((s + 1) - (int(skip_last)-1)):  # changed from s + 1

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations

                n_configs = n * self.eta ** (-i)
                n_iterations = r * self.eta ** (i)


                if (self.counter+1)>(self.max_iter/(1-1/self.eta)-1):
                    break


                print("\n*** {} configurations x {:.1f} hyperband iterations each".format(
                    n_configs, n_iterations))

                val_losses = []
                val_accuracies = []
                early_stops = []


                j = -1
                for t in T:
                    j += 1
                    self.counter += 1

                    print("\n* {} | {} \n".format(
                        self.counter, ctime()))

                    start_time = time()

                    if dry_run:
                        result = {'loss': random(), 'log_loss': random(), 'auc': random()}
                    else:

                        result = self.try_params(n_iterations, t, fixed_params, data_train, data_validate)  # <---
                        ## make sure we test 81 models ##
                        while result['loss'] > 1e6:
                            try:
                                t = self.get_params()
                                result = self.try_params(n_iterations, t, fixed_params, data_train, data_validate)
                                T[j] = t
                            except Exception:
                                print("Error")
                                result['loss'] = np.inf
                    assert (type(result) == dict)
                    assert ('loss' in result or 'acc' in result)

                    seconds = int(round(time() - start_time))
                    print("\n{} seconds.".format(seconds))






                    # keeping track of the best result so far (for display only)
                    # could do it be checking results each time, but hey
                    if 'loss' in result:
                        loss = result['loss']
                        val_losses.append(loss)
                        if loss < self.best_loss:
                            self.best_loss = loss
                            self.best_counter = self.counter
                    if 'acc' in result:
                        accuracy = result['acc']
                        val_accuracies.append(accuracy)
                        if accuracy > self.best_accuracy:
                            self.best_accuracy = accuracy
                            self.best_counter = self.counter


                    print("\n{} | {} | lowest loss so far: {:.4f} (run {})\n".format(
                        self.counter, ctime(), self.best_loss, self.best_counter))
                    print("*****************************************************")


                    early_stop = result.get('early_stop', False)
                    early_stops.append(early_stop)



                    result['counter'] = self.counter
                    result['seconds'] = seconds
                    result['params'] = t
                    result['iterations'] = n_iterations

                    self.results.append(result)

                # select a number of best configurations (= highest accuracies) for the next loop
                # filter out early stops, if any
#                indices_of_accuracies_ascending_order = np.argsort(val_accuracies)
#                indices_of_highest_accuracies_first = indices_of_accuracies_ascending_order[::-1]
#                T = [T[i] for i in indices_of_highest_accuracies_first if not early_stops[i]]

                indices_of_loss_ascending_order = np.argsort(val_losses)
                indices_of_highest_loss_first = indices_of_loss_ascending_order[::-1]
                T = [T[i] for i in indices_of_loss_ascending_order if not early_stops[i]]
                T = T[0:int(n_configs / self.eta)]

        return self.results
