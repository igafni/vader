import multiprocessing
import dask
from multiprocessing.pool import ThreadPool

from utils.utils_functions import split_list_by_list_num, split_list_by_items_num_per_list


class ThreadRunner(object):
    def __init__(self, num_of_threads):
        self.num_of_threads = num_of_threads
        self.jobs = []

    @staticmethod
    def run_target_with_dask(target, target_kwargs: list):
        delayed_predictions = [dask.delayed(target)(input_data) for input_data in target_kwargs]
        predictions = dask.compute(*delayed_predictions)
        return predictions

    def run_target_in_threading(self, target, target_kwargs: list):
        target_split = split_list_by_list_num(target_kwargs, self.num_of_threads)
        pool = ThreadPool(processes=self.num_of_threads)
        return pool.map(target, target_split)

    def run_target_in_processing(self, target, target_kwargs: list):
        target_split = split_list_by_list_num(target_kwargs, self.num_of_threads)
        pool = multiprocessing.Pool(processes=self.num_of_threads)
        data = pool.map(target, target_split)
        pool.close()
        return data

    def run_target_in_processing2(self, target, target_kwargs: list):
        target_split = split_list_by_list_num(target_kwargs, self.num_of_threads)
        with multiprocessing.Pool(processes=self.num_of_threads) as pool:
            multiprocessing.freeze_support()
            data = pool.map(target, target_split)
            multiprocessing.freeze_support()
            return data

    def run_target_in_processing_3(self, target, target_kwargs: list):
        target_split = split_list_by_list_num(target_kwargs, self.num_of_threads)
        for target_list in target_split:
            process = multiprocessing.Process(
                target=target,
                args=(target_list)
            )
            self.jobs.append(process)
        for j in self.jobs:
            j.start()

        for j in self.jobs:
            j.join()
        return self.jobs

    # pool = multiprocessing.Process(processes=self.num_of_threads)
    # return pool.map(target, target_split)

# for splits in target_split:
#     async_results = [pool.apply_async(target, args=[kwarg, *tuple(kwargs.values())]) for kwarg in splits]
# return [result.get() for result in async_results]
