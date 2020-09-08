''' moderation_analysis.py A model for analysis of the effect of NESA HSC moderation on student marks

'''

import numpy as np
import matplotlib

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

def adjust_exam(school, exam):
    # first sort then
    # if school scores are tied, recalculate the maximum exam score
    exam_sorted = np.flip(np.sort(exam))
    tied_places = 0
    for i in range(np.size(school)):
        if school[0] == school[i]:
            tied_places = i
    
    if tied_places:
        high_score = np.rint(np.sum(exam_sorted[:tied_places+1])/(tied_places + 1))
        exam_sorted[:tied_places+1] = high_score
        # replace_above = exam_sorted[tied_places] - 1
        # print(high_score, replace_above)
        
        # for i in range(np.size(exam)):
        #     if exam[i] > replace_above:
        #         exam[i] = np.rint(high_score)
        

    return exam_sorted
    
def stretch(in_array, out_min, out_max):
    low = np.min(in_array)
    high = np.max(in_array)
    in_range = high - low
    return out_min + (in_array - low)* (out_max - out_min)/in_range    
 

def moderate(school, exam):
    school = stretch(school, np.min(exam), np.max(exam))
    school = np.rint(school)
    mod_poly = np.polyfit(np.sort(school), np.sort(exam), 2)
    mod_marks = mod_poly[0]*np.power(school, 2)+ mod_poly[1]*school + mod_poly[2]
    # mod_marks = np.rint(mod_marks)
    # print(f'before stretch {mod_marks}')
    # mod_marks = stretch(mod_marks, np.min(exam), np.max(exam))
    mod_marks = np.rint(mod_marks)
    # print(f'after stretch {mod_marks}')
    return mod_marks

def get_stats(in_array):
    return np.array([np.amin(in_array), np.rint(np.mean(in_array)), np.amax(in_array), np.sum(in_array)])


if __name__ == "__main__":
    n = 6
    mean = 65
    sd = 15
    mean_delta = 4
    sd_delta = 4
    school_max = 95
    nesa_check = False

    if nesa_check:
        school = np.array([90, 78, 75, 58, 55, 40])
        exam  = np.array([92, 72, 80, 60, 50, 55])
        nesa = np.array([92, 77, 74, 59, 57, 50])

        moderated = moderate(school, exam)

        print(f'The school assessment marks:                {np.rint(school)}')
        print(f'The exam marks:                             {np.rint(exam)}')
        print(f'The moderated marks from the NESA web page: {np.rint(nesa)}')
        print(f'The moderated marks from this algorithm:    {moderated}')
        print(f'stats for exam [min, mean, max]:            {get_stats(exam)}')
        print(f'stats from nesa [min, mean, max]:           {get_stats(nesa)}')
        print(f'stats for moderated marks [min, mean, max]: {get_stats(moderated)}')

    else:
    
        school_scores = generate_class_scores(n, mean, sd, max_score=school_max)
        exam_scores = generate_exam_scores(school_scores, mean_delta, sd_delta)

        print(f'school scores:    {school_scores}')
        print(f'exam scores:      {exam_scores}')
        
        exam_adjusted = adjust_exam(school_scores, exam_scores)

        print(f'adjusted exam:    {exam_adjusted}')

        moderated_scores = moderate(school_scores, exam_adjusted)
        print(f'moderated scores: {moderated_scores}')

        print(f'school scores stats:    {get_stats(school_scores)}')
        print(f'exam scores stats:      {get_stats(exam_scores)}')
        print(f'adjusted exam stats:    {get_stats(exam_adjusted)}')
        print(f'moderated scores stats: {get_stats(moderated_scores)}')


print(stretch([2,4,5,8], 8, 32))