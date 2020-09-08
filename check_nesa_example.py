import numpy as np
school = np.array([90, 78, 75, 58, 55, 40])
exam  = np.array([92, 72, 80, 60, 50, 55])
nesa = np.rint(np.array([92, 77, 74, 59, 57, 50]))

def get_stats(in_array, tied = False):
    if tied:
        # in the case of tied rankings NESA averages the two top exam marks before fitting the polynomial
        # need to calculate this tied mark before calculating poly_fit_tied
        in_array = np.sort(in_array)
        top_mark = np.rint((in_array[n-2] + in_array[n-1])/2)
        stats = np.array([np.amin(in_array), np.rint(np.mean(in_array)), top_mark])
    else:
        stats = np.array([np.amin(in_array), np.rint(np.mean(in_array)), np.amax(in_array)])
    return stats

stats_school = get_stats(school)
stats_exam = get_stats(exam)

poly_mod = np.polyfit(stats_school, stats_exam, 2)

stats_check = poly_mod[0]*np.power(stats_school, 2)+ poly_mod[1]*stats_school + poly_mod[2]
print(f'stats check: {stats_check}')

moderated = poly_mod[0]*np.power(school, 2)+ poly_mod[1]*school + poly_mod[2]
moderated = np.rint(moderated)

print(f'The moderated marks from the NESA web page: {nesa}')
print(f'The moderated marks from this algorithm:    {moderated}')
print(f'stats for exam [min, mean, max]:            {stats_exam}, total: {np.sum(exam)}')
print(f'stats from nesa [min, mean, max]:           {get_stats(nesa)}, total: {np.sum(nesa)}')
print(f'stats for moderated marks [min, mean, max]: {get_stats(moderated)}, total: {np.sum(moderated)}')
