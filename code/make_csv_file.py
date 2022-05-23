import numpy as np
import os
import csv

test_result = np.load('../result/pred_seq2seq.npy')
result_path = '../result'

def output2csv(pred_y, file_name=os.path.join(result_path, 'sent_class.pred.csv')):
    os.makedirs(result_path, exist_ok=True)
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'pred'])
        for i, p in enumerate(pred_y):
            y_id = str(i + 1)
            if len(y_id) < 5:
                y_id = '0' * (5 - len(y_id)) + y_id
            writer.writerow(['S'+y_id, p])
    print('file saved.')

output2csv(test_result)