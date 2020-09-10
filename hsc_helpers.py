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

def multi_run_plot(effect, n, runs):
    ranks = np.arange(1,n+1)
    plt.figure()
    plt.errorbar(ranks, np.mean(effect, axis=0), yerr=np.std(effect, axis=0), fmt='o')
    plt.axhline(0, color='black')
    plt.xlabel('school rank')
    plt.ylabel(f'mean & sd of benefit over {runs} runs')
    plt.title('Effect of separating tied first placegetters')
    plt.show()

def multi_run(n=15, mean=65, sd=15, mean_delta=4, sd_delta=5, school_max=95, runs=1000, tied='tie'):
    
    winners_losers = np.empty(n)
    
    for ctr in range(1, runs):
        school_scores = generate_class_scores(n, mean, sd, max_score=school_max, tied='tie')
        school_scores_split = copy.deepcopy(school_scores)
        school_scores_split[0] += 1

        exam_scores = generate_exam_scores(school_scores, mean_delta, sd_delta, n)
        
        moderated_scores = moderate(school_scores, exam_scores)
        moderated_scores_split = moderate(school_scores_split, exam_scores)

        # this is painful but numpy need to append like to like so...
        # see if you can spot the difference
        if ctr != 1:
            winners_losers = np.append(winners_losers, [(moderated_scores_split - moderated_scores)], axis = 0)
        else: # first run
            winners_losers = np.append([winners_losers], [(moderated_scores_split - moderated_scores)], axis = 0)
            
    return winners_losers
    
    multi_run_plot(winners_losers, n, runs)

