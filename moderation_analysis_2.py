''' moderation_analysis.py A model for analysis of the effect of NESA HSC moderation on student marks

'''

import numpy as np
import matplotlib
import copy

def test_min_max(in_array, min_limit, max_limit):
    return ((np.min(in_array) >= min_limit) and (np.max(in_array) <= max_limit))


def test_tied_condition(in_array, tied):
    ''' test whether the scores are tied as specified.
    '''
    # if ((tied == 'tie') or (tied =='pair') or (tied =='triple')):
    if tied in ('tie', 'pair', 'triple'):
        if tied == 'pair':
            return ((in_array[0] == in_array[1]) and (in_array[1] != in_array[2]))
        elif tied == 'triple':
            return (in_array[0] == in_array[2])
        else:
            return (in_array[0] == in_array[1])


    else:
        return (in_array[0] != in_array[1])


def generate_class_scores(n, mean, sd, min_score= 0, max_score= 100, tied='normal'):
    '''Generates randomly distributed class scores.

    Args:
        n (int > 3): number of students in the class
        mean (int): the arithmetic mean of the scores
        sd (float): the standard distribution of the scores
        mimin_scoren (int): the minimum allowable score
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


def generate_exam_scores(scores, mean_delta, sd_delta):
    conditions_met = False

    while not conditions_met:
        exam = np.rint(scores + np.random.normal(mean_delta, sd_delta, n))
        conditions_met = test_min_max(exam, 0, 100)
        
    return exam

def check_ties(school, exam):
    # first sort then
    # if school scores are tied, recalculate the maximum exam score
    exam_checked = np.flip(np.sort(exam))
    n = np.size(school) - 1
    tied_firsts = 0
    tied_lasts = 0
    for i in range(1,n):
        if school[0] == school[i]:
            tied_firsts = i
        if school[n] == school[n-i]:
            tied_lasts = i
    
    if tied_firsts:
        high_score = np.sum(exam_checked[:tied_firsts+1])/(tied_firsts + 1)
        exam_checked[:tied_firsts+1] = high_score
        print(f'{tied_firsts} tied firsts, marks adjusted: {exam_checked}')

    if tied_lasts:
        low_score = np.sum(exam_checked[n-tied_lasts:])/(tied_lasts + 1)
        exam_checked[n-tied_lasts:] = low_score
        print(f'{tied_lasts} tied lasts, marks adjusted: {exam_checked}')


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

        # ya normal case
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
            

if __name__ == "__main__":
    n = 15
    mean = 65
    sd = 15
    mean_delta = 4
    sd_delta = 5
    school_max = 95
    nesa_check = False

    if nesa_check:
        school = np.array([90, 78, 75, 58, 55, 40])
        exam  = np.array([92, 72, 80, 60, 50, 55])
        nesa = np.array([92, 77, 74, 59, 57, 50])

        print(np.polyfit(school,nesa,2))

        moderated = moderate(school, exam)

        print(f'The school assessment marks:                {np.rint(school)}')
        print(f'The exam marks:                             {np.rint(exam)}')
        print(f'The moderated marks from the NESA web page: {np.rint(nesa)}')
        print(f'The moderated marks from this algorithm:    {moderated}')
        print(f'stats for exam [max, min, mean, sd, sum]:            {np.rint(get_stats(exam))}')
        print(f'stats from nesa [max, min, mean, sd, sum]:           {np.rint(get_stats(nesa))}')
        print(f'stats for moderated marks [max, min, mean, sd, sum]: {np.rint(get_stats(moderated))}')

    else:
    
        school_scores = generate_class_scores(n, mean, sd, max_score=school_max, tied='tie')
        school_scores_split = copy.deepcopy(school_scores)
        school_scores_split[0] += 1

        exam_scores = generate_exam_scores(school_scores, mean_delta, sd_delta)

        print(f'school scores:          {school_scores}')
        print(f'school scores split:    {school_scores_split}')
        print(f'exam scores:            {exam_scores}')
        
        moderated_scores = moderate(school_scores, exam_scores)
        moderated_scores_split = moderate(school_scores_split, exam_scores)
        print(f'moderated scores:       {moderated_scores}')
        print(f'moderated scores split: {moderated_scores_split}')

        print(f'school scores stats:          {np.rint(get_stats(school_scores))}')
        print(f'exam scores stats:            {np.rint(get_stats(exam_scores))}')
        print(f'moderated scores stats:       {np.rint(get_stats(moderated_scores))}')
        print(f'moderated scores split stats: {np.rint(get_stats(moderated_scores))}')
        print(f'winners and losers:           {moderated_scores_split - moderated_scores}')


test = np.array([1,3,3, 3,1,5])
# test = np.unique(test)
print(test)
print(np.size(np.unique(test)))