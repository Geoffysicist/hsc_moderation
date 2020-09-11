import numpy as np
import matplotlib.pyplot as plt
import copy

def test_min_max(in_array, min_limit, max_limit):
    '''Checks whether a numpy array is within defined limits.

    Returns:
        True if within the bounds, otherwise False
    '''
    return ((np.min(in_array) >= min_limit) and (np.max(in_array) <= max_limit))


def test_tied_condition(in_array, tied):
    ''' test whether the scores are tied as specified.

    Args:
        in_array (numpy array) - array to be checked
        tied (string) - a string describing the different tied states.
                valid values are:
                    'normal' no restriction on tied scores
                    'none' tied scores not allowed
                    'pair' exactly 2 tied scores required
                    'triple' exactly 3 tied scores required
                    'tie' if 2 or more tied scores required
    Returns:
        True if scores are tied as specified above, otherwise False
    '''
    
    if tied in ('tie', 'pair', 'triple'):
        if tied == 'pair':
            return ((in_array[0] == in_array[1]) and (in_array[1] != in_array[2]))
        elif tied == 'triple':
            return (in_array[0] == in_array[2])
        else:
            return (in_array[0] == in_array[1])
    elif tied == 'none':
        return (in_array[0] != in_array[1])
    else:
        return True


def generate_class_scores(n, mean, sd, min_score= 0, max_score= 100, tied='normal'):
    '''Generates randomly distributed class scores.

    Keeps generating random sets of scores until one is generated meeting the
    conditions specified by min, max and tied

    Args:.
        n (int > 3): number of students in the class
        mean (int): the arithmetic mean of the scores
        sd (float): the standard distribution of the scores
        min_score (int): the minimum allowable score
        max_score (int): the maximum allowable score
        tied (str): 'normal' no restriction on tied scores,
                    'none' tied scores not allowed
                    'pair' exactly 2 tied scores required
                    'triple' exactly 3 tied scores required
                    'tie' if 2 or more tied scores required

    Returns:
        class_scores (numpy array): the class scores.
    '''

    conditions_met = False

    while not conditions_met:

        class_scores = np.random.normal(mean, sd, n)
        class_scores = np.flip(np.sort(class_scores)) # order highest to lowest
        class_scores = np.rint(class_scores) 

        conditions_met = test_min_max(class_scores, min_score, max_score)

        if conditions_met:
            conditions_met = test_tied_condition(class_scores, tied)

    return class_scores


def generate_exam_scores(scores, mean_delta, sd_delta, n):
    '''Generates exam scores by adding a random number to school scores.

    Each score is varied independently by a random amount.
    Repeats if exam scores are not between 0 and 100.

    Args:
        scores (numpy array) - the school scores
        mean_delta (int) - the mean amount to add to the score. Scores will decrease
            on average if this is set to negative.
        sd_delta (float) - the standard deviation of the amount ot add to the scores

    Returns:
        exam (numpy array) - exam scores
    '''

    conditions_met = False

    while not conditions_met:
        exam = np.rint(scores + np.random.normal(mean_delta, sd_delta, n))
        conditions_met = test_min_max(exam, 0, 100)
        
    return exam

def check_ties(school, exam):
    '''Adjusts exam scores if top or bottom school marks are tied.

    If there are n tied school scores, the top exam score is set to the 
    mean of the top n exam scores. A similar adjustment is made for tied bottom scores. 

    Args:
        school (numpy array) - school assessment scores
        exam (numpy array) - exam scores

    Returns:
        exam_checked (numpy array) - sorted, and if required adjusted exam scores
    '''
    exam_checked = np.flip(np.sort(exam)) # sort into descending order
    n = np.size(school) 
    tied_firsts = 0
    tied_lasts = 0
    for i in range(1,n-1): # array indices start at 0
        if school[0] == school[i]:
            tied_firsts = i+1
        if school[n-1] == school[(n-1)-i]:
            tied_lasts = i+1
    
    if tied_firsts:
        high_score = np.sum(exam_checked[:tied_firsts])/(tied_firsts)
        exam_checked[:tied_firsts] = high_score
        
    if tied_lasts:
        low_score = np.sum(exam_checked[n-tied_lasts:])/(tied_lasts)
        exam_checked[n-tied_lasts:] = low_score
        
    return exam_checked
 
def get_stats(in_array):
    ''' Calculate the descriptive stats of a numpy array

    Args:
        in_array (numpy array) - data to be described

    Returns:
        stats_array (numpy array) - an array containing the max, min, mean
            standard deviation and sum of the input array
    '''
    stats_array = np.array([np.amax(in_array),
                    np.amin(in_array), 
                    np.mean(in_array), 
                    np.std(in_array), 
                    np.sum(in_array)])

    return stats_array

def moderate(school, exam):
    ''' moderate school scores as described by MacCann 1995.

    MacCann RG 1995 'The Moderation of Higher School Certificate Assessments using a 
        Quadratic Polynomial Transformation: a Technical Paper', Board of Studies NSW.

    Args:
        school (numpy array) - school assessment scores
        exam (numpy array) - exam scores

    Returns:
        moderated (numpy array) - moderated school scores    
    '''

    school_stats = get_stats(school)
    school_max = school_stats[0]
    school_min = school_stats[1]
    school_mean = school_stats[2]
    school_sd = school_stats[3]
    exam_stats = get_stats(exam)
    exam_max = exam_stats[0]
    exam_min = exam_stats[1]
    exam_mean = exam_stats[2]
    n = np.size(school)

    # start with special cases
    # n = 1
    if n == 1:
        moderated = exam

    # n = 2
    elif n == 2:
        mod_min = max(exam_min, (school_min/school_max)*exam_max)
        moderated = np.array([exam_max, mod_min])
    
    # all school ranks tied
    elif school_max == school_min:
        moderated = np.mean(exam)

    else:
        # n > 2 but only 2 distinct values
        if np.size(np.unique(school)) == 2:
            a = 0
            b = (exam_max - exam_min) / (school_max - school_min)
            c = exam_min - b * school_min

        # finally ya normal boi!
        else:
            # TODO make this more elegant
            exam = check_ties(school, exam)
            exam_stats = get_stats(exam)
            exam_max = exam_stats[0]
            exam_min = exam_stats[1]
            exam_mean = exam_stats[2]
    
            a_num = exam_max*(school_min-school_mean) - exam_min*(school_max-school_mean) + exam_mean*(school_max-school_min)
            a_denom = (school_max-school_min)*(school_sd**2 + (school_max-school_mean)*(school_min-school_mean))
            a = a_num/a_denom

            b = (exam_min - exam_mean - a*(school_min**2-school_mean**2-school_sd**2))/(school_min-school_mean)

            c = exam_min - school_min*(a*school_min + b)
        
        moderated = np.rint(np.polyval([a,b,c],school))

    return moderated

def multi_run_plot(diff_mod_cum, diff_hsc_cum, n, params_text):

    ranks = np.arange(1,n+1)
    
    rank_title_text = 'Effect of moderation of school scores by rank'

    fig = plt.figure(figsize=(8,6))

    plt.errorbar(ranks, np.mean(diff_mod_cum, axis=0), yerr=np.std(diff_mod_cum, axis=0), marker='X', label='moderated score', c=(0,0.45,0.7), capsize=6, ls='')
    plt.errorbar(ranks, np.mean(diff_hsc_cum, axis=0), yerr=np.std(diff_hsc_cum, axis=0), marker='P', label='hsc mark', c=(0.9,0.6,0), capsize=6, ls='')
    plt.legend()       
    plt.axhline(0, color='lightgrey', lw=1)
    plt.xlabel('school rank\n\n' + params_text)
    plt.ylabel(f'mean & sd of moderated:hsc score minus school score')
    plt.title(rank_title_text)
     
    plt.show()
    
def multi_run_plot_split(diff_mod_cum, diff_hsc_cum, diff_mod_split_cum, diff_hsc_split_cum, n, tied, params_text):
    # diff_mod_cum, diff_hsc_cum, diff_mod_split_cum, diff_hsc_split_cum, n, tied, params_text
    
    ranks = np.arange(1,n+1)
    
    #compare the split and tied cases
    diff_split_tied = diff_mod_split_cum - diff_mod_cum
    
    if tied == 'pair':
        tied_state = 'exactly two'
    elif tied == 'triple':
        tied_state = 'exactly three'
    else:
        tied_state = 'at least two'
    
    fig = plt.figure(figsize=(16,6))

    plt.subplot(1, 2, 1)
    plt.errorbar(ranks, np.mean(diff_mod_cum, axis=0), yerr=np.std(diff_mod_cum, axis=0), marker='X', label='score tied', c=(0,0.45,0.7), capsize=6, ls='')
    plt.errorbar(ranks, np.mean(diff_mod_split_cum, axis=0), yerr=np.std(diff_mod_split_cum, axis=0), marker='P', label='score split', c=(0.9,0.6,0), capsize=6, ls='')
    plt.legend()       
    plt.axhline(0, color='lightgrey', lw=1)
    plt.xlabel('school rank\n\n' + params_text)
    plt.ylabel(f'mean & sd of tied:split score minus school score')
    plt.title(f'Comparison of moderation of tied and split school scores\nwhen {tied_state} top scores are tied\nby rank')
    
    plt.subplot(1, 2, 2)
    plt.errorbar(ranks, np.mean(diff_split_tied, axis=0), yerr=np.std(diff_split_tied, axis=0), marker='P', c=(0,0.45,0.7), capsize=6, ls='')
    plt.axhline(0, color='lightgrey', lw=1)
    plt.xlabel('school rank\n\n' + params_text)
    plt.ylabel(f'mean & sd of split minus tied score')
    plt.title(f'Difference between moderated split and tied school scores\nen {tied_state} top scores are tied\nby rank')
    
    plt.show()
    
def single_run_plot_split(n, school_scores, exam_scores, mod_scores, mod_scores_split, hsc_scores, params_text):
    ranks = np.arange(1,n+1)
    
    fig = plt.figure(figsize=(8,6))
    
    marker_size = 100

    plt.scatter(ranks, school_scores, label='school score', color=(0,0.45,0.7), marker='+', s=marker_size)
    plt.scatter(ranks, exam_scores, label='exam mark', color=(0.9,0.6,0), marker='x', s=marker_size)
    plt.scatter(ranks, mod_scores, label='moderated score tied', color=(0.8,0.4,0), marker='3', s=marker_size)
    plt.scatter(ranks, mod_scores_split, label='moderated score split', color=(0,0.6,0.5), marker='4', s=marker_size)
    plt.legend()       
    plt.xlabel('school rank\n\n' + params_text)
    plt.ylabel(f'mean & sd of moderated:hsc score minus school score')
    plt.title('Comparison of school scores, exam scores and moderated scores')
     
    plt.show()
    
def single_run_plot(n, school_scores, exam_scores, mod_scores, hsc_scores, params_text):
    ranks = np.arange(1,n+1)
    
    fig = plt.figure(figsize=(8,6))
    
    marker_size = 100

    plt.scatter(ranks, school_scores, label='school score', color=(0,0.45,0.7), marker='+', s=marker_size)
    plt.scatter(ranks, exam_scores, label='exam mark', color=(0.9,0.6,0), marker='1', s=marker_size)
    plt.scatter(ranks, mod_scores, label='moderated score', color=(0.8,0.4,0.4), marker='x', s=marker_size)
    plt.legend()       
    plt.xlabel('school rank\n\n' + params_text)
    plt.ylabel(f'mean & sd of moderated:hsc score minus school score')
    plt.title('Comparison of school scores, exam scores and moderated scores')
     
    plt.show()
    

def multi_run(inputs):
    [n, mean, sd, school_max, mean_delta, sd_delta, runs, tied] = inputs
    
    params_text = (
            f'Model parameters:\n'
            f'    number of students in the subject at the school = {n}\n'
            f'    mean of the school assessment scores = {mean}\n'
            f'    standard deviation of the school assessment scores = {sd}\n'
            f'    maximum school mark = {school_max}\n'
            f'    mean of the difference between school assessment and hsc scores = {mean_delta}\n'
            f'    standard deviation of the difference between school assessment and hsc scores = {sd_delta}\n'
            f'    number of data sets or iterations compared = {runs}.'
            )

    diff_mod_cum = np.empty(n)
    diff_hsc_cum = np.empty(n)
    
    if tied in ('tie', 'pair', 'triple'):
        split = True
        diff_mod_split_cum = np.empty(n)
        diff_hsc_split_cum = np.empty(n)
    else:
        split = False
    
    
    for ctr in range(runs):
        school_scores = generate_class_scores(n, mean, sd, max_score=school_max, tied='tie')
        exam_scores = generate_exam_scores(school_scores, mean_delta, sd_delta, n)
        
        mod_scores = moderate(school_scores, exam_scores)
        hsc_scores = np.mean(np.array([mod_scores, exam_scores]), axis=0)
        diff_mod = mod_scores - school_scores
        diff_hsc = hsc_scores - school_scores
            
        if split:
            # the tied cases
            school_scores_split = copy.deepcopy(school_scores)
            school_scores_split[0] += 1
            mod_scores_split = moderate(school_scores_split, exam_scores)
            hsc_scores_split = np.mean(np.array([mod_scores_split, exam_scores]), axis=0)
            diff_mod_split = mod_scores_split - school_scores
            diff_hsc_split = hsc_scores_split - school_scores

        # this is painful but numpy need to append like to like so...
        # see if you can spot the difference
        if ctr != 0:
            diff_mod_cum = np.append(diff_mod_cum, [diff_mod], axis = 0)
            diff_hsc_cum = np.append(diff_hsc_cum, [diff_hsc], axis = 0)
            if split:
                diff_mod_split_cum = np.append(diff_mod_split_cum, [diff_mod_split], axis = 0)
                diff_hsc_split_cum = np.append(diff_hsc_split_cum, [diff_hsc_split], axis = 0)
        else: # first run
            diff_mod_cum = np.append([diff_mod_cum], [diff_mod], axis = 0)
            diff_hsc_cum = np.append([diff_hsc_cum], [diff_hsc], axis = 0)
            if split: 
                diff_mod_split_cum = np.append([diff_mod_split_cum], [diff_mod_split], axis = 0)
                diff_hsc_split_cum = np.append([diff_hsc_split_cum], [diff_hsc_split], axis = 0)
    
    # remove the first empty rows
    diff_mod_cum = np.delete(diff_mod_cum,0,0)
    diff_hsc_cum = np.delete(diff_hsc_cum,0,0)
    
    
    if split:
        # remove the first empty rows
        diff_mod_split_cum = np.delete(diff_mod_split_cum,0,0)
        diff_hsc_split_cum = np.delete(diff_hsc_split_cum,0,0)
        
        if runs == 1:
            single_run_plot_split(n, school_scores, exam_scores, mod_scores, mod_scores_split, hsc_scores, params_text)            
        else:
            multi_run_plot_split(diff_mod_cum, diff_hsc_cum, diff_mod_split_cum, diff_hsc_split_cum, n, tied, params_text)

    else:
        if runs == 1:
            single_run_plot(n, school_scores, exam_scores, mod_scores, hsc_scores, params_text) 
        else:
            multi_run_plot(diff_mod_cum, diff_hsc_cum, n, params_text)

