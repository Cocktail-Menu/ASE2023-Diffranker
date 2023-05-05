import os
MAX_LEN = 120

if __name__ == '__main__':
    
    for dataset, split in [('KB13.bak', 'val'), ('Turk.bak', 'test')]:

        NL_path = os.path.join(dataset, 'src-{}.txt'.format(split))
        ground_truth_path = os.path.join(dataset, 'targ-{}.txt'.format(split))
        with open(NL_path, 'r') as f:
            NLs = f.read()
            NLs = NLs.replace("<M0>", '"dog"')
            NLs = NLs.replace("<M1>", '"truck"')
            NLs = NLs.replace("<M2>", '"ring"')
            NLs = NLs.replace("<M3>", '"lake"')
            NLs = NLs.splitlines()

        with open(ground_truth_path, 'r') as f:
            ground_truths = f.read()
            ground_truths = ground_truths.replace(" ", "")
            ground_truths = ground_truths.splitlines()

        # NL_path = os.path.join(dataset.split('.')[0], 'src-{}.txt'.format(split))
        # ground_truth_path = os.path.join(dataset.split('.')[0], 'targ-{}.txt'.format(split))
        # with open(NL_path, 'w') as a, open(ground_truth_path, 'w') as b:
        #     count = 0
        #     for NL, ground_truth in zip(NLs, ground_truths):
        #         if '&' not in ground_truth:
        #             count += 1
        #             if count < MAX_LEN:
        #                 a.write(NL + '\n')
        #                 b.write(ground_truth + '\n')
        #             else:
        #                 a.write(NL)
        #                 b.write(ground_truth)
        #                 break
        #     print(count)

        count = 0
        for NL, ground_truth in zip(NLs, ground_truths):
            if '&' not in ground_truth:
                count += 1
        print(count, len(NLs), count / len(NLs))
        


