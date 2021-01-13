import zlib


def compress(uncompressed_string):
    """An implementation of the Lempel-Ziv-Welch compression algorithm with variable-width codes

    Parameters
    ----------
    uncompressed_string

    Returns
    -------
    a list of output symbols
    """

    # build the dictionary
    dict_size = 256
    dictionary = dict((chr(i), i) for i in range(dict_size))

    w = ""
    compressed_values = []
    for c in uncompressed_string:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            # add w to the output
            compressed_values.append(dictionary[w])
            # add wc to the dictionary
            dictionary[wc] = dict_size
            dict_size += 1
            w = c

    if w:
        compressed_values.append(dictionary[w])

    return compressed_values


# count bits assuming algorithm uses variable-width codes
def calculate_bits(compressed_values):

    size = len(compressed_values)
    total_bits = 0

    # characters are stored in 9-12 bits if compressed
    # (characters are stored in 8 bits if uncompressed)
    # if we need to move up a code width level, then we continue to use that
    # code width level until we have to move up again
    code_width = 0
    for output in compressed_values:
        if (output >= 0) and (output < 512) and (code_width < 9):
            code_width = 9
        if (output >= 512) and (output < 1024) and (code_width < 10):
            code_width = 10
        if (output >= 1024) and (output < 2048) and (code_width < 11):
            code_width = 11
        if (output >= 2048) and (code_width < 12):
            code_width = 12
        total_bits += code_width

    return total_bits


# calculate an approximation for K(x) given a string x
# given a string x, the Kolmogorov complexity K(x) is the minimum number of bits
# into which the string can be compressed without losing information
def approximate_KC_string(x):
    compressed_string = compress(x)
    total_bits = calculate_bits(compressed_string)
    return total_bits


# calculate an approximation for K(xy) given string x and string y
def approximate_KC_concat(x, y):
    concat = x + y
    compressed_string = compress(concat)
    total_bits = calculate_bits(compressed_string)
    return total_bits


# calculate an approximation for K(xy), which is the denominator in Normalized Information Distance
def approximate_KC_concat_for_normalization(x, y):
    total_bits_1 = approximate_KC_concat(x, y)
    total_bits_2 = approximate_KC_concat(y, x)
    average = float(total_bits_1 + total_bits_2) / 2
    return average


# calculate an approximation for K(x|y) by calculating K(x|y) = K(xy) - K(y)
def approxiate_KC_conditional(x, y):
    KC_concat = approximate_KC_concat(x, y)
    KC_y = approximate_KC_string(y)
    KC_conditional = KC_concat - KC_y
    return KC_conditional


# calculate an approximation for NID = (K(x|y) + K(y|x)) / K(xy)
# two identical sequences will have NID = 0
# two sequences with no common information will have NID = 1
# but, we a performing an approximation of NID based on Kolmogorov complexity,
# so the NID returned from this function will only give us an upper bound
def approximate_NID(x, y):
    KC_conditional_xy = approxiate_KC_conditional(x, y)
    KC_conditional_yx = approxiate_KC_conditional(y, x)
    KC_concat_normalization = approximate_KC_concat_for_normalization(x, y)
    NID = float(KC_conditional_xy + KC_conditional_yx) / KC_concat_normalization
    # print "\nK(x|y) = %f, K(y|x) = %f, K(xy) = %f" % (KC_conditional_xy, KC_conditional_yx, KC_concat_normalization)
    return NID


def approximate_NID_v2(x, y):
    KC_conditional_xy = approxiate_KC_conditional(x, y)
    KC_conditional_yx = approxiate_KC_conditional(y, x)
    KC_x = approximate_KC_string(x)
    KC_y = approximate_KC_string(y)
    NID = (float(max(KC_conditional_xy, KC_conditional_yx))) / (max(KC_x, KC_y))
    # print "\nK(x|y) = %f, K(y|x) = %f, K(x) = %f, K(y) = %f" % (KC_conditional_xy, KC_conditional_yx, KC_x, KC_y)
    return NID


# Реализуем NCD приближение-метрики
def NCD(x, y):
    # Сжимаем строку x и находим длину сжатой версии
    Cx = zlib.compress(x)
    Cx_len = len(Cx)
    # Сжимаем строку y и находим длину сжатой версии
    Cy = zlib.compress(y)
    Cy_len = len(Cy)

    # конкатенируем строки x и y, сжимаем их и находим длину
    xy = x + y
    Cxy = zlib.compress(xy)
    Cxy_len = len(Cxy)

    # рассчитываем искомое расстояние по NCD формуле
    distance = float(Cxy_len - min(Cx_len, Cy_len)) / max(Cx_len, Cy_len)

    return distance
