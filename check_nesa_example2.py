import numpy as np
school = np.array([90, 78, 75, 58, 55, 40])
exam  = np.array([92, 72, 80, 60, 50, 55])
nesa = np.array([92, 77, 74, 59, 57, 50])

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

def stretch(in_array, out_min, out_max):
    low = np.min(in_array)
    high = np.max(in_array)
    in_range = high - low
    return out_min + (in_array - low)* (out_max - out_min)/in_range

def moderate(school_marks, exam_marks):
    school_marks_ordered = np.flip(np.sort(school_marks))
    exam_marks_ordered = np.flip(np.sort(exam_marks))

    print(school_marks_ordered, exam_marks_ordered)

    # check for tied first place in school assessment
    if school_marks_ordered[0] == school_marks_ordered[1]
        exam_top = (exam_marks_ordered[0]+exam_marks_ordered[1])/2
        

    mod_poly = np.polyfit(school_marks_ordered, exam_marks_ordered, 2)
    mod_marks = mod_poly[0]*np.power(school_marks, 2)+ mod_poly[1]*school_marks + mod_poly[2]
    mod_marks = stretch(mod_marks, np.min(exam), np.max(exam))
    mod_marks = np.rint(mod_marks)

    return mod_marks

moderated = moderate(school, exam)

print(f'school assessment marks:                {np.rint(school)}')
print(f'exam marks:                             {np.rint(exam)}')
print(f'moderated marks from the NESA web page: {np.rint(nesa)}')
print(f'moderated marks from this algorithm:    {moderated}')
print(f'stats for exam [min, mean, max]:            {get_stats(exam)}, total: {np.sum(exam)}')
print(f'stats from nesa [min, mean, max]:           {get_stats(nesa)}, total: {np.sum(nesa)}')
print(f'stats for moderated marks [min, mean, max]: {get_stats(moderated)}, total: {np.sum(moderated)}')
