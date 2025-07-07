
# This function ends up being used by a lot of things, just based on how the math plays out.
# keeping it in its own file keeps the code organized. 

def single_axis_positioning(a, i, q, d, t):

    return (a / 60) * (i + q / d)- (q * t / (60 * d))