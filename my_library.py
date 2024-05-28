def test_load():
  return 'loaded'
def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]
  
def cond_prob(table, evidence, evidence_value, target, target_value):
  t_subset = up_table_subset(table, target, 'equals', target_value)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a + .01

def cond_probs_product(table, evidence_row, target, target_value):
  evidence_complete = up_zip_lists(table.columns[:-1], evidence_row)
  cond_prob_list = [cond_prob(table, evi[0], evi[1], target, target_value) for evi in evidence_complete]
  partial_numerator = up_product(cond_prob_list)
  return partial_numerator

def prior_prob(table, target, target_value):
  t_list = up_get_column(table, target)
  p_a = sum([1 if v==target_value else 0 for v in t_list])/len(t_list)
  return p_a

def naive_bayes(table, evidence_row, target):
  cond_prob_no = cond_probs_product(table, evidence_row, target, 0)
  prior_prob_no = prior_prob(table, target, 0)
  prob_target_no = cond_prob_no * prior_prob_no

  cond_prob_yes = cond_probs_product(table, evidence_row, target, 1)
  prior_prob_yes = prior_prob(table, target, 1)
  prob_target_yes = cond_prob_yes * prior_prob_yes

  neg, pos = compute_probs(prob_target_no, prob_target_yes)
  return [neg, pos]

def metrics(zipped_list):

  assert isinstance(zipped_list, list), f'Parameter is not a list'
  assert all([isinstance(i, list) for i in zipped_list]), f'Parameter is not a list of lists'
  assert all([len(i) == 2 for i in zipped_list]), f'Parameter is not a zipped list - one or more values is not a pair of items'
  assert all([(i[0] >= 0 and i[1] >= 0) and (isinstance(i[0], int) and isinstance(i[1], int)) for i in zipped_list]), f'One or more values in a pair is either a non-integer or a negative number'


  tn = sum([1 if pair==[0,0] else 0 for pair in zipped_list])
  tp = sum([1 if pair==[1,1] else 0 for pair in zipped_list])
  fp = sum([1 if pair==[1,0] else 0 for pair in zipped_list])
  fn = sum([1 if pair==[0,1] else 0 for pair in zipped_list])

  precision = (tp) / (tp + fp) if (tp + fp) else 0
  recall = (tp) / (tp + fn) if (tp + fn) else 0
  f1 = 2*((precision * recall) / (precision + recall)) if (precision + recall) else 0
  accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0

  results = {'Precision': precision, 'Recall': recall, 'F1': f1, 'Accuracy': accuracy}

  return results
