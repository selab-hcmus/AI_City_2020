"""Make metrics for BEAT-PD"""
import numpy as np
class BeatPDMetrics():
    def __init__(self):
        self.result_dict = {}
        self.max_number = 10000.0

    def add_results(self, results, labels, sid_lst):
        """Add new testing results

        Args:
            results:
            sid_lst:
        """
        assert len(results) == len(sid_lst), 'Mismatching lists'
        N = len(results)

        for i in range(N):
            res = self.max_number if np.isnan(results[i]) else results[i]
            lbl = labels[i]
            sid = sid_lst[i]

            if sid not in(self.result_dict):
                self.result_dict[sid] = {
                    'res': [],
                    'lbl': [],
                }
            self.result_dict[sid]['res'].append(res)
            self.result_dict[sid]['lbl'].append(lbl)

    def clean_data(self):
        """Clean up result dictionary
        """
        self.result_dict.clear()

    def compute_score(self):
        """Compute the final score

        More information of the metrics is found here:
        https://www.synapse.org/#!Synapse:syn20825169/wiki/600897

        Return:
            Computed final MSE score
        """
        # Allocate memory
        mse_lst = np.zeros(len(self.result_dict))
        len_lst = np.zeros(len(self.result_dict))

        # Go through all subjects
        for i, sid in enumerate(self.result_dict):
            # Compute the MSE of the current subject
            res = np.array(self.result_dict[sid]['res'])
            lbl = np.array(self.result_dict[sid]['lbl'])
           
            mse = np.mean((res - lbl) ** 2)
            # Concat the MSE and number of testing observations per subject
            mse_lst[i] = mse
            len_lst[i] = len(res)

        # Compute the final score
        sqrt_len = np.sqrt(len_lst)
        final_score = np.sum(sqrt_len * mse_lst) / np.sum(sqrt_len)
        return final_score
