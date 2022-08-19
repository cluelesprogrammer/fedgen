import numpy as np
import yaml


def sample_multi_label_distribution(batch_size, label_p):
    #labels are drawn with a prior probability, atleast one label per sample is drawn
    #label_p needs to be multiplied by two because the datasets are really sparse
    label_p = np.array(list(label_p.values())) * 1.5
    all_samples = []
    for i in range(batch_size):
      while (True):
        sampled_y = np.random.binomial(1, label_p)
        least_one_label_bool = np.array([np.any(sampled_y == 1)])
        if (least_one_label_bool):
          all_samples.append(sampled_y)
          break
    return np.stack(all_samples)


"""
def sample_multi_label_distribution(label_p, batch_size):

def sample_multi_label_distribution(batch_size, label_p):
    #labels are drawn with a prior probability, atleast one label per sample is drawn
    #label_p needs to be multiplied by two because the datasets are really sparse 
    label_p = np.array(list(label_p.values())) * 2
    while (True):
      sampled_y = np.random.binomial(1, np.tile(label_p, batch_size)).reshape(
                  batch_size, len(label_p))
      least_one_label_bool = np.array([np.any(y == 1) for y in sampled_y])
      least_one_label_per_sample = np.all(least_one_label_bool)
      if (least_one_label_per_sample):
        break
    return sampled_y
    #labels are drawn with a probability higher than it iwereif it were uniformly distributed
    #to encourage multi label samples
    while True:
        if qualified_labels=='all':
            prior_label_distribution = np.tile(np.array(2.5/total_labels), total_labels)
            sampled_y = np.random.binomial(1, np.tile(prior_label_distribution, batch_size)).reshape(
                batch_size, total_labels)
        elif (type(qualified_labels) == list):
            prior_label_distribution = np.zeros(total_labels)
            prior_label_distribution[qualified_labels] = label_p
            prior_label_distribution = np.tile(prior_label_distribution, batch_size)
            sampled_y = np.random.binomial(1, prior_label_distribution).reshape(batch_size, total_labels)
        else:
            print('wrong input')
        #to ensure that at least one labels has been sampled
        least_one_label_bool = np.array([np.any(y == 1) for y in sampled_y])
        least_one_label_per_sample = np.all(least_one_label_bool)
        if (least_one_label_per_sample):
            break
    return sampled_y
"""
