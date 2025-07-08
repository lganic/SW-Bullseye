import math

# This function ends up being used by a lot of things, just based on how the math plays out.
def single_axis_positioning(a, i, q, d, t):

    return (a / 60) * (i + q / d)- (q * t / (60 * d))

# Again, used by a bunch of things
def split_to_vector(direction: float, magnitude: float):
    '''
    Direction in degrees. 
    '''

    return magnitude * math.cos(math.radians(direction)), magnitude * math.sin(math.radians(direction))