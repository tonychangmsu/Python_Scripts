#If I am to write the code myself, I can just copy the solar_position function and work from there....
#import libraries
import numpy as np
import pandas as pd
import datetime as datetime
import pytz as pytz
import calendar as calendar 

#Constants table from 
TABLE_1_DICT = {
    'L0': np.array(
        [[175347046.0, 0.0, 0.0],
         [3341656.0, 4.6692568, 6283.07585],
         [34894.0, 4.6261, 12566.1517],
         [3497.0, 2.7441, 5753.3849],
         [3418.0, 2.8289, 3.5231],
         [3136.0, 3.6277, 77713.7715],
         [2676.0, 4.4181, 7860.4194],
         [2343.0, 6.1352, 3930.2097],
         [1324.0, 0.7425, 11506.7698],
         [1273.0, 2.0371, 529.691],
         [1199.0, 1.1096, 1577.3435],
         [990.0, 5.233, 5884.927],
         [902.0, 2.045, 26.298],
         [857.0, 3.508, 398.149],
         [780.0, 1.179, 5223.694],
         [753.0, 2.533, 5507.553],
         [505.0, 4.583, 18849.228],
         [492.0, 4.205, 775.523],
         [357.0, 2.92, 0.067],
         [317.0, 5.849, 11790.629],
         [284.0, 1.899, 796.298],
         [271.0, 0.315, 10977.079],
         [243.0, 0.345, 5486.778],
         [206.0, 4.806, 2544.314],
         [205.0, 1.869, 5573.143],
         [202.0, 2.458, 6069.777],
         [156.0, 0.833, 213.299],
         [132.0, 3.411, 2942.463],
         [126.0, 1.083, 20.775],
         [115.0, 0.645, 0.98],
         [103.0, 0.636, 4694.003],
         [102.0, 0.976, 15720.839],
         [102.0, 4.267, 7.114],
         [99.0, 6.21, 2146.17],
         [98.0, 0.68, 155.42],
         [86.0, 5.98, 161000.69],
         [85.0, 1.3, 6275.96],
         [85.0, 3.67, 71430.7],
         [80.0, 1.81, 17260.15],
         [79.0, 3.04, 12036.46],
         [75.0, 1.76, 5088.63],
         [74.0, 3.5, 3154.69],
         [74.0, 4.68, 801.82],
         [70.0, 0.83, 9437.76],
         [62.0, 3.98, 8827.39],
         [61.0, 1.82, 7084.9],
         [57.0, 2.78, 6286.6],
         [56.0, 4.39, 14143.5],
         [56.0, 3.47, 6279.55],
         [52.0, 0.19, 12139.55],
         [52.0, 1.33, 1748.02],
         [51.0, 0.28, 5856.48],
         [49.0, 0.49, 1194.45],
         [41.0, 5.37, 8429.24],
         [41.0, 2.4, 19651.05],
         [39.0, 6.17, 10447.39],
         [37.0, 6.04, 10213.29],
         [37.0, 2.57, 1059.38],
         [36.0, 1.71, 2352.87],
         [36.0, 1.78, 6812.77],
         [33.0, 0.59, 17789.85],
         [30.0, 0.44, 83996.85],
         [30.0, 2.74, 1349.87],
         [25.0, 3.16, 4690.48]]),
    'L1': np.array(
        [[628331966747.0, 0.0, 0.0],
         [206059.0, 2.678235, 6283.07585],
         [4303.0, 2.6351, 12566.1517],
         [425.0, 1.59, 3.523],
         [119.0, 5.796, 26.298],
         [109.0, 2.966, 1577.344],
         [93.0, 2.59, 18849.23],
         [72.0, 1.14, 529.69],
         [68.0, 1.87, 398.15],
         [67.0, 4.41, 5507.55],
         [59.0, 2.89, 5223.69],
         [56.0, 2.17, 155.42],
         [45.0, 0.4, 796.3],
         [36.0, 0.47, 775.52],
         [29.0, 2.65, 7.11],
         [21.0, 5.34, 0.98],
         [19.0, 1.85, 5486.78],
         [19.0, 4.97, 213.3],
         [17.0, 2.99, 6275.96],
         [16.0, 0.03, 2544.31],
         [16.0, 1.43, 2146.17],
         [15.0, 1.21, 10977.08],
         [12.0, 2.83, 1748.02],
         [12.0, 3.26, 5088.63],
         [12.0, 5.27, 1194.45],
         [12.0, 2.08, 4694.0],
         [11.0, 0.77, 553.57],
         [10.0, 1.3, 6286.6],
         [10.0, 4.24, 1349.87],
         [9.0, 2.7, 242.73],
         [9.0, 5.64, 951.72],
         [8.0, 5.3, 2352.87],
         [6.0, 2.65, 9437.76],
         [6.0, 4.67, 4690.48]]),
    'L2': np.array(
        [[52919.0, 0.0, 0.0],
         [8720.0, 1.0721, 6283.0758],
         [309.0, 0.867, 12566.152],
         [27.0, 0.05, 3.52],
         [16.0, 5.19, 26.3],
         [16.0, 3.68, 155.42],
         [10.0, 0.76, 18849.23],
         [9.0, 2.06, 77713.77],
         [7.0, 0.83, 775.52],
         [5.0, 4.66, 1577.34],
         [4.0, 1.03, 7.11],
         [4.0, 3.44, 5573.14],
         [3.0, 5.14, 796.3],
         [3.0, 6.05, 5507.55],
         [3.0, 1.19, 242.73],
         [3.0, 6.12, 529.69],
         [3.0, 0.31, 398.15],
         [3.0, 2.28, 553.57],
         [2.0, 4.38, 5223.69],
         [2.0, 3.75, 0.98]]),
    'L3': np.array(
        [[289.0, 5.844, 6283.076],
         [35.0, 0.0, 0.0],
         [17.0, 5.49, 12566.15],
         [3.0, 5.2, 155.42],
         [1.0, 4.72, 3.52],
         [1.0, 5.3, 18849.23],
         [1.0, 5.97, 242.73]]),
    'L4': np.array(
        [[114.0, 3.142, 0.0],
         [8.0, 4.13, 6283.08],
         [1.0, 3.84, 12566.15]]),
    'L5': np.array(
        [[1.0, 3.14, 0.0]]),
    'B0': np.array(
        [[280.0, 3.199, 84334.662],
         [102.0, 5.422, 5507.553],
         [80.0, 3.88, 5223.69],
         [44.0, 3.7, 2352.87],
         [32.0, 4.0, 1577.34]]),
    'B1': np.array(
        [[9.0, 3.9, 5507.55],
         [6.0, 1.73, 5223.69]]),
    'R0': np.array(
        [[100013989.0, 0.0, 0.0],
         [1670700.0, 3.0984635, 6283.07585],
         [13956.0, 3.05525, 12566.1517],
         [3084.0, 5.1985, 77713.7715],
         [1628.0, 1.1739, 5753.3849],
         [1576.0, 2.8469, 7860.4194],
         [925.0, 5.453, 11506.77],
         [542.0, 4.564, 3930.21],
         [472.0, 3.661, 5884.927],
         [346.0, 0.964, 5507.553],
         [329.0, 5.9, 5223.694],
         [307.0, 0.299, 5573.143],
         [243.0, 4.273, 11790.629],
         [212.0, 5.847, 1577.344],
         [186.0, 5.022, 10977.079],
         [175.0, 3.012, 18849.228],
         [110.0, 5.055, 5486.778],
         [98.0, 0.89, 6069.78],
         [86.0, 5.69, 15720.84],
         [86.0, 1.27, 161000.69],
         [65.0, 0.27, 17260.15],
         [63.0, 0.92, 529.69],
         [57.0, 2.01, 83996.85],
         [56.0, 5.24, 71430.7],
         [49.0, 3.25, 2544.31],
         [47.0, 2.58, 775.52],
         [45.0, 5.54, 9437.76],
         [43.0, 6.01, 6275.96],
         [39.0, 5.36, 4694.0],
         [38.0, 2.39, 8827.39],
         [37.0, 0.83, 19651.05],
         [37.0, 4.9, 12139.55],
         [36.0, 1.67, 12036.46],
         [35.0, 1.84, 2942.46],
         [33.0, 0.24, 7084.9],
         [32.0, 0.18, 5088.63],
         [32.0, 1.78, 398.15],
         [28.0, 1.21, 6286.6],
         [28.0, 1.9, 6279.55],
         [26.0, 4.59, 10447.39]]),
    'R1': np.array(
        [[103019.0, 1.10749, 6283.07585],
         [1721.0, 1.0644, 12566.1517],
         [702.0, 3.142, 0.0],
         [32.0, 1.02, 18849.23],
         [31.0, 2.84, 5507.55],
         [25.0, 1.32, 5223.69],
         [18.0, 1.42, 1577.34],
         [10.0, 5.91, 10977.08],
         [9.0, 1.42, 6275.96],
         [9.0, 0.27, 5486.78]]),
    'R2': np.array(
        [[4359.0, 5.7846, 6283.0758],
         [124.0, 5.579, 12566.152],
         [12.0, 3.14, 0.0],
         [9.0, 3.63, 77713.77],
         [6.0, 1.87, 5573.14],
         [3.0, 5.47, 18849.23]]),
    'R3': np.array(
        [[145.0, 4.273, 6283.076],
         [7.0, 3.92, 12566.15]]),
    'R4': np.array(
        [[4.0, 2.56, 6283.08]])
}
TABLE_1_DICT['L1'].resize((64, 3))
TABLE_1_DICT['L2'].resize((64, 3))
TABLE_1_DICT['L3'].resize((64, 3))
TABLE_1_DICT['L4'].resize((64, 3))
TABLE_1_DICT['L5'].resize((64, 3))

TABLE_1_DICT['B1'].resize((5, 3))

TABLE_1_DICT['R1'].resize((40, 3))
TABLE_1_DICT['R2'].resize((40, 3))
TABLE_1_DICT['R3'].resize((40, 3))
TABLE_1_DICT['R4'].resize((40, 3))

HELIO_LONG_TABLE = np.array([TABLE_1_DICT['L0'],
                             TABLE_1_DICT['L1'],
                             TABLE_1_DICT['L2'],
                             TABLE_1_DICT['L3'],
                             TABLE_1_DICT['L4'],
                             TABLE_1_DICT['L5']])

HELIO_LAT_TABLE = np.array([TABLE_1_DICT['B0'],
                            TABLE_1_DICT['B1']])

HELIO_RADIUS_TABLE = np.array([TABLE_1_DICT['R0'],
                               TABLE_1_DICT['R1'],
                               TABLE_1_DICT['R2'],
                               TABLE_1_DICT['R3'],
                               TABLE_1_DICT['R4']])

NUTATION_ABCD_ARRAY = np.array([
    [-171996, -174.2, 92025, 8.9],
    [-13187, -1.6, 5736, -3.1],
    [-2274, -0.2, 977, -0.5],
    [2062, 0.2, -895, 0.5],
    [1426, -3.4, 54, -0.1],
    [712, 0.1, -7, 0],
    [-517, 1.2, 224, -0.6],
    [-386, -0.4, 200, 0],
    [-301, 0, 129, -0.1],
    [217, -0.5, -95, 0.3],
    [-158, 0, 0, 0],
    [129, 0.1, -70, 0],
    [123, 0, -53, 0],
    [63, 0, 0, 0],
    [63, 0.1, -33, 0],
    [-59, 0, 26, 0],
    [-58, -0.1, 32, 0],
    [-51, 0, 27, 0],
    [48, 0, 0, 0],
    [46, 0, -24, 0],
    [-38, 0, 16, 0],
    [-31, 0, 13, 0],
    [29, 0, 0, 0],
    [29, 0, -12, 0],
    [26, 0, 0, 0],
    [-22, 0, 0, 0],
    [21, 0, -10, 0],
    [17, -0.1, 0, 0],
    [16, 0, -8, 0],
    [-16, 0.1, 7, 0],
    [-15, 0, 9, 0],
    [-13, 0, 7, 0],
    [-12, 0, 6, 0],
    [11, 0, 0, 0],
    [-10, 0, 5, 0],
    [-8, 0, 3, 0],
    [7, 0, -3, 0],
    [-7, 0, 0, 0],
    [-7, 0, 3, 0],
    [-7, 0, 3, 0],
    [6, 0, 0, 0],
    [6, 0, -3, 0],
    [6, 0, -3, 0],
    [-6, 0, 3, 0],
    [-6, 0, 3, 0],
    [5, 0, 0, 0],
    [-5, 0, 3, 0],
    [-5, 0, 3, 0],
    [-5, 0, 3, 0],
    [4, 0, 0, 0],
    [4, 0, 0, 0],
    [4, 0, 0, 0],
    [-4, 0, 0, 0],
    [-4, 0, 0, 0],
    [-4, 0, 0, 0],
    [3, 0, 0, 0],
    [-3, 0, 0, 0],
    [-3, 0, 0, 0],
    [-3, 0, 0, 0],
    [-3, 0, 0, 0],
    [-3, 0, 0, 0],
    [-3, 0, 0, 0],
    [-3, 0, 0, 0],
])

NUTATION_YTERM_ARRAY = np.array([
    [0, 0, 0, 0, 1],
    [-2, 0, 0, 2, 2],
    [0, 0, 0, 2, 2],
    [0, 0, 0, 0, 2],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [-2, 1, 0, 2, 2],
    [0, 0, 0, 2, 1],
    [0, 0, 1, 2, 2],
    [-2, -1, 0, 2, 2],
    [-2, 0, 1, 0, 0],
    [-2, 0, 0, 2, 1],
    [0, 0, -1, 2, 2],
    [2, 0, 0, 0, 0],
    [0, 0, 1, 0, 1],
    [2, 0, -1, 2, 2],
    [0, 0, -1, 0, 1],
    [0, 0, 1, 2, 1],
    [-2, 0, 2, 0, 0],
    [0, 0, -2, 2, 1],
    [2, 0, 0, 2, 2],
    [0, 0, 2, 2, 2],
    [0, 0, 2, 0, 0],
    [-2, 0, 1, 2, 2],
    [0, 0, 0, 2, 0],
    [-2, 0, 0, 2, 0],
    [0, 0, -1, 2, 1],
    [0, 2, 0, 0, 0],
    [2, 0, -1, 0, 1],
    [-2, 2, 0, 2, 2],
    [0, 1, 0, 0, 1],
    [-2, 0, 1, 0, 1],
    [0, -1, 0, 0, 1],
    [0, 0, 2, -2, 0],
    [2, 0, -1, 2, 1],
    [2, 0, 1, 2, 2],
    [0, 1, 0, 2, 2],
    [-2, 1, 1, 0, 0],
    [0, -1, 0, 2, 2],
    [2, 0, 0, 2, 1],
    [2, 0, 1, 0, 0],
    [-2, 0, 2, 2, 2],
    [-2, 0, 1, 2, 1],
    [2, 0, -2, 0, 1],
    [2, 0, 0, 0, 1],
    [0, -1, 1, 0, 0],
    [-2, -1, 0, 2, 1],
    [-2, 0, 0, 0, 1],
    [0, 0, 2, 2, 1],
    [-2, 0, 2, 0, 1],
    [-2, 1, 0, 2, 1],
    [0, 0, 1, -2, 0],
    [-1, 0, 1, 0, 0],
    [-2, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 1, 2, 0],
    [0, 0, -2, 2, 2],
    [-1, -1, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, -1, 1, 2, 2],
    [2, -1, -1, 2, 2],
    [0, 0, 3, 2, 2],
    [2, -1, 0, 2, 2],
])


def julian_day(unixtime):
    jd = unixtime * 1.0 / 86400 + 2440587.5
    return(jd)

def julian_ephemeris_day(julian_day, delta_t):
    jde = julian_day + delta_t * 1.0 / 86400
    return(jde)

def julian_century(julian_day):
    jc = (julian_day - 2451545) * 1.0 / 36525
    return(jc)

def julian_ephemeris_century(julian_ephemeris_day):
    jce = (julian_ephemeris_day - 2451545) * 1.0 / 36525
    return(jce)

def julian_ephemeris_millennium(julian_ephemeris_century):
    jme = julian_ephemeris_century * 1.0 / 10
    return(jme)

def heliocentric_longitude(jme):
    l0 = 0.0
    l1 = 0.0
    l2 = 0.0
    l3 = 0.0
    l4 = 0.0
    l5 = 0.0

    for row in range(HELIO_LONG_TABLE.shape[1]):
        l0 += (HELIO_LONG_TABLE[0, row, 0]
               * np.cos(HELIO_LONG_TABLE[0, row, 1]
                        + HELIO_LONG_TABLE[0, row, 2] * jme)
               )
        l1 += (HELIO_LONG_TABLE[1, row, 0]
               * np.cos(HELIO_LONG_TABLE[1, row, 1]
                        + HELIO_LONG_TABLE[1, row, 2] * jme)
               )
        l2 += (HELIO_LONG_TABLE[2, row, 0]
               * np.cos(HELIO_LONG_TABLE[2, row, 1]
                        + HELIO_LONG_TABLE[2, row, 2] * jme)
               )
        l3 += (HELIO_LONG_TABLE[3, row, 0]
               * np.cos(HELIO_LONG_TABLE[3, row, 1]
                        + HELIO_LONG_TABLE[3, row, 2] * jme)
               )
        l4 += (HELIO_LONG_TABLE[4, row, 0]
               * np.cos(HELIO_LONG_TABLE[4, row, 1]
                        + HELIO_LONG_TABLE[4, row, 2] * jme)
               )
        l5 += (HELIO_LONG_TABLE[5, row, 0]
               * np.cos(HELIO_LONG_TABLE[5, row, 1]
                        + HELIO_LONG_TABLE[5, row, 2] * jme)
               )

    l_rad = (l0 + l1 * jme + l2 * jme**2 + l3 * jme**3 + l4 * jme**4 +
             l5 * jme**5)/10**8
    l = np.rad2deg(l_rad)
    return(l % 360)
	
def heliocentric_latitude(jme):
    b0 = 0.0
    b1 = 0.0
    for row in range(HELIO_LAT_TABLE.shape[1]):
        b0 += (HELIO_LAT_TABLE[0, row, 0]
               * np.cos(HELIO_LAT_TABLE[0, row, 1]
                        + HELIO_LAT_TABLE[0, row, 2] * jme)
               )
        b1 += (HELIO_LAT_TABLE[1, row, 0]
               * np.cos(HELIO_LAT_TABLE[1, row, 1]
                        + HELIO_LAT_TABLE[1, row, 2] * jme)
               )

    b_rad = (b0 + b1 * jme)/10**8
    b = np.rad2deg(b_rad)
    return(b)

def heliocentric_radius_vector(jme):
    r0 = 0.0
    r1 = 0.0
    r2 = 0.0
    r3 = 0.0
    r4 = 0.0
    for row in range(HELIO_RADIUS_TABLE.shape[1]):
        r0 += (HELIO_RADIUS_TABLE[0, row, 0]
               * np.cos(HELIO_RADIUS_TABLE[0, row, 1]
                        + HELIO_RADIUS_TABLE[0, row, 2] * jme)
               )
        r1 += (HELIO_RADIUS_TABLE[1, row, 0]
               * np.cos(HELIO_RADIUS_TABLE[1, row, 1]
                        + HELIO_RADIUS_TABLE[1, row, 2] * jme)
               )
        r2 += (HELIO_RADIUS_TABLE[2, row, 0]
               * np.cos(HELIO_RADIUS_TABLE[2, row, 1]
                        + HELIO_RADIUS_TABLE[2, row, 2] * jme)
               )
        r3 += (HELIO_RADIUS_TABLE[3, row, 0]
               * np.cos(HELIO_RADIUS_TABLE[3, row, 1]
                        + HELIO_RADIUS_TABLE[3, row, 2] * jme)
               )
        r4 += (HELIO_RADIUS_TABLE[4, row, 0]
               * np.cos(HELIO_RADIUS_TABLE[4, row, 1]
                        + HELIO_RADIUS_TABLE[4, row, 2] * jme)
               )

    r = (r0 + r1 * jme + r2 * jme**2 + r3 * jme**3 + r4 * jme**4)/10**8
    return(r)

def geocentric_longitude(heliocentric_longitude):
    theta = heliocentric_longitude + 180.0
    return(theta % 360)
	
def geocentric_latitude(heliocentric_latitude):
    beta = -1.0*heliocentric_latitude
    return(beta)

def mean_elongation(julian_ephemeris_century):
    x0 = (297.85036
          + 445267.111480 * julian_ephemeris_century
          - 0.0019142 * julian_ephemeris_century**2
          + julian_ephemeris_century**3 / 189474)
    return(x0)

def mean_anomaly_sun(julian_ephemeris_century):
    x1 = (357.52772
          + 35999.050340 * julian_ephemeris_century
          - 0.0001603 * julian_ephemeris_century**2
          - julian_ephemeris_century**3 / 300000)
    return(x1)

def mean_anomaly_moon(julian_ephemeris_century):
    x2 = (134.96298
          + 477198.867398 * julian_ephemeris_century
          + 0.0086972 * julian_ephemeris_century**2
          + julian_ephemeris_century**3 / 56250)
    return(x2)

def moon_argument_latitude(julian_ephemeris_century):
    x3 = (93.27191
          + 483202.017538 * julian_ephemeris_century
          - 0.0036825 * julian_ephemeris_century**2
          + julian_ephemeris_century**3 / 327270)
    return(x3)

def moon_ascending_longitude(julian_ephemeris_century):
    x4 = (125.04452
          - 1934.136261 * julian_ephemeris_century
          + 0.0020708 * julian_ephemeris_century**2
          + julian_ephemeris_century**3 / 450000)
    return(x4)

def longitude_nutation(julian_ephemeris_century, x0, x1, x2, x3, x4):
    delta_psi_sum = 0
    for row in range(NUTATION_YTERM_ARRAY.shape[0]):
        a = NUTATION_ABCD_ARRAY[row, 0]
        b = NUTATION_ABCD_ARRAY[row, 1]
        argsin = (NUTATION_YTERM_ARRAY[row, 0]*x0 +
                  NUTATION_YTERM_ARRAY[row, 1]*x1 +
                  NUTATION_YTERM_ARRAY[row, 2]*x2 +
                  NUTATION_YTERM_ARRAY[row, 3]*x3 +
                  NUTATION_YTERM_ARRAY[row, 4]*x4)
        term = (a + b * julian_ephemeris_century) * np.sin(np.radians(argsin))
        delta_psi_sum += term
    delta_psi = delta_psi_sum*1.0/36000000
    return(delta_psi)

def obliquity_nutation(julian_ephemeris_century, x0, x1, x2, x3, x4):
    delta_eps_sum = 0.0
    for row in range(NUTATION_YTERM_ARRAY.shape[0]):
        c = NUTATION_ABCD_ARRAY[row, 2]
        d = NUTATION_ABCD_ARRAY[row, 3]
        argcos = (NUTATION_YTERM_ARRAY[row, 0]*x0 +
                  NUTATION_YTERM_ARRAY[row, 1]*x1 +
                  NUTATION_YTERM_ARRAY[row, 2]*x2 +
                  NUTATION_YTERM_ARRAY[row, 3]*x3 +
                  NUTATION_YTERM_ARRAY[row, 4]*x4)
        term = (c + d * julian_ephemeris_century) * np.cos(np.radians(argcos))
        delta_eps_sum += term
    delta_eps = delta_eps_sum*1.0/36000000
    return(delta_eps)

def mean_ecliptic_obliquity(julian_ephemeris_millennium):
    U = 1.0*julian_ephemeris_millennium/10
    e0 = (84381.448 - 4680.93 * U - 1.55 * U**2
          + 1999.25 * U**3 - 51.38 * U**4 - 249.67 * U**5
          - 39.05 * U**6 + 7.12 * U**7 + 27.87 * U**8
          + 5.79 * U**9 + 2.45 * U**10)
    return(e0)

def true_ecliptic_obliquity(mean_ecliptic_obliquity, obliquity_nutation):
    e0 = mean_ecliptic_obliquity
    deleps = obliquity_nutation
    e = e0*1.0/3600 + deleps
    return(e)

def aberration_correction(earth_radius_vector):
    deltau = -20.4898 / (3600 * earth_radius_vector)
    return(deltau)

def apparent_sun_longitude(geocentric_longitude, longitude_nutation,
                           aberration_correction):
    lamd = geocentric_longitude + longitude_nutation + aberration_correction
    return(lamd)

def mean_sidereal_time(julian_day, julian_century):
    v0 = (280.46061837 + 360.98564736629 * (julian_day - 2451545)
          + 0.000387933 * julian_century**2 - julian_century**3 / 38710000)
    return(v0 % 360.0)

def apparent_sidereal_time(mean_sidereal_time, longitude_nutation,
                           true_ecliptic_obliquity):
    v = mean_sidereal_time + longitude_nutation * np.cos(
        np.radians(true_ecliptic_obliquity))
    return(v)

def geocentric_sun_right_ascension(apparent_sun_longitude,
                                   true_ecliptic_obliquity,
                                   geocentric_latitude):
    num = (np.sin(np.radians(apparent_sun_longitude))
           * np.cos(np.radians(true_ecliptic_obliquity))
           - np.tan(np.radians(geocentric_latitude))
           * np.sin(np.radians(true_ecliptic_obliquity)))
    alpha = np.degrees(np.arctan2(num, np.cos(
        np.radians(apparent_sun_longitude))))
    return(alpha % 360)

def geocentric_sun_declination(apparent_sun_longitude, true_ecliptic_obliquity,
                               geocentric_latitude):
    delta = np.degrees(np.arcsin(np.sin(np.radians(geocentric_latitude)) *
                                 np.cos(np.radians(true_ecliptic_obliquity)) +
                                 np.cos(np.radians(geocentric_latitude)) *
                                 np.sin(np.radians(true_ecliptic_obliquity)) *
                                 np.sin(np.radians(apparent_sun_longitude))))
    return(delta)

def local_hour_angle(apparent_sidereal_time, observer_longitude, sun_right_ascension):
	"""Measured westward from south"""
	H = apparent_sidereal_time + observer_longitude - sun_right_ascension
	#H = np.add(apparent_sidereal_time + observer_longitude) - sun_right_ascension
	return(H % 360)

def equatorial_horizontal_parallax(earth_radius_vector):
    xi = 8.794 / (3600 * earth_radius_vector)
    return(xi)

def uterm(observer_latitude):
    u = np.arctan(0.99664719 * np.tan(np.radians(observer_latitude)))
    return(u)

def xterm(u, observer_latitude, observer_elevation):
    x = (np.cos(u) + observer_elevation / 6378140
         * np.cos(np.radians(observer_latitude)))
    return(x)

def yterm(u, observer_latitude, observer_elevation):
    y = (0.99664719 * np.sin(u) + observer_elevation / 6378140
         * np.sin(np.radians(observer_latitude)))
    return(y)

def parallax_sun_right_ascension(xterm, equatorial_horizontal_parallax,
                                 local_hour_angle, geocentric_sun_declination):
    num = (-xterm * np.sin(np.radians(equatorial_horizontal_parallax))
           * np.sin(np.radians(local_hour_angle)))
    denom = (np.cos(np.radians(geocentric_sun_declination))
             - xterm * np.sin(np.radians(equatorial_horizontal_parallax))
             * np.cos(np.radians(local_hour_angle)))
    delta_alpha = np.degrees(np.arctan2(num, denom))
    return(delta_alpha)

def topocentric_sun_right_ascension(geocentric_sun_right_ascension,
                                    parallax_sun_right_ascension):
    alpha_prime = geocentric_sun_right_ascension + parallax_sun_right_ascension
    return(alpha_prime)

def topocentric_sun_declination(geocentric_sun_declination, xterm, yterm,
                                equatorial_horizontal_parallax,
                                parallax_sun_right_ascension,
                                local_hour_angle):
    num = ((np.sin(np.radians(geocentric_sun_declination)) - yterm
            * np.sin(np.radians(equatorial_horizontal_parallax)))
           * np.cos(np.radians(parallax_sun_right_ascension)))
    denom = (np.cos(np.radians(geocentric_sun_declination)) - xterm
             * np.sin(np.radians(equatorial_horizontal_parallax))
             * np.cos(np.radians(local_hour_angle)))
    delta = np.degrees(np.arctan2(num, denom))
    return(delta)
	
def topocentric_local_hour_angle(local_hour_angle,
                                 parallax_sun_right_ascension):
    H_prime = local_hour_angle - parallax_sun_right_ascension
    return(H_prime)

def topocentric_elevation_angle_without_atmosphere(observer_latitude,
                                                   topocentric_sun_declination,
                                                   topocentric_local_hour_angle
                                                   ):
    e0 = np.degrees(np.arcsin(
        np.sin(np.radians(observer_latitude))
        * np.sin(np.radians(topocentric_sun_declination))
        + np.cos(np.radians(observer_latitude))
        * np.cos(np.radians(topocentric_sun_declination))
        * np.cos(np.radians(topocentric_local_hour_angle))))
    return(e0)

def atmospheric_refraction_correction(local_pressure, local_temp,
                                      topocentric_elevation_angle_wo_atmosphere,
                                      atmos_refract):
    # switch sets delta_e when the sun is below the horizon
    switch = topocentric_elevation_angle_wo_atmosphere >= -1.0 * (
        0.26667 + atmos_refract)
    delta_e = ((local_pressure / 1010.0) * (283.0 / (273 + local_temp))
               * 1.02 / (60 * np.tan(np.radians(
                   topocentric_elevation_angle_wo_atmosphere
                   + 10.3 / (topocentric_elevation_angle_wo_atmosphere
                             + 5.11))))) * switch
    return(delta_e)

def topocentric_elevation_angle(topocentric_elevation_angle_without_atmosphere,
                                atmospheric_refraction_correction):
    e = (topocentric_elevation_angle_without_atmosphere
         + atmospheric_refraction_correction)
    return(e)

def topocentric_zenith_angle(topocentric_elevation_angle):
    theta = 90 - topocentric_elevation_angle
    return theta

def topocentric_astronomers_azimuth(topocentric_local_hour_angle,
                                    topocentric_sun_declination,
                                    observer_latitude):
    num = np.sin(np.radians(topocentric_local_hour_angle))
    denom = (np.cos(np.radians(topocentric_local_hour_angle))
             * np.sin(np.radians(observer_latitude))
             - np.tan(np.radians(topocentric_sun_declination))
             * np.cos(np.radians(observer_latitude)))
    gamma = np.degrees(np.arctan2(num, denom))
    return(gamma % 360)

def topocentric_azimuth_angle(topocentric_astronomers_azimuth):
    phi = topocentric_astronomers_azimuth + 180
    return(phi % 360)

def sun_mean_longitude(julian_ephemeris_millennium):
    M = (280.4664567 + 360007.6982779 * julian_ephemeris_millennium
         + 0.03032028 * julian_ephemeris_millennium**2
         + julian_ephemeris_millennium**3 / 49931
         - julian_ephemeris_millennium**4 / 15300
         - julian_ephemeris_millennium**5 / 2000000)
    return(M)

def equation_of_time(sun_mean_longitude, geocentric_sun_right_ascension,
                     longitude_nutation, true_ecliptic_obliquity):
    E = (sun_mean_longitude - 0.0057183 - geocentric_sun_right_ascension +
         longitude_nutation * np.cos(np.radians(true_ecliptic_obliquity)))
    # limit between 0 and 360
    E = E % 360
    # convert to minutes
    E *= 4
    greater = E > 20
    less = E < -20
    other = (E <= 20) & (E >= -20)
    E = greater * (E - 1440) + less * (E + 1440) + other * E
    return(E)

def unix_time_convert(year, month, day, local_tz):
	date = '%s/%s/%s'%(year, month, day)
	naive = datetime.datetime.strptime("%s"%(date), "%Y/%m/%d")
	local_dt = local_tz.localize(naive) #assign the time zone
	utc_dt = local_dt.astimezone(pytz.utc) 
	#this is correct (6 hour difference)
	unixtime_local = calendar.timegm(local_dt.timetuple())
	unixtime_utc = calendar.timegm(utc_dt.timetuple()) #this is one hour off from the current unix time. why!?
	return(unixtime_utc)
	
#other constants

#use bozeman lat lon and elevation 
lat = 45.67965 #45 40 46.74
lon = -111.038560 #-111 2 18.816
elev = 1469.
now = datetime.datetime.now()


local_tz = pytz.timezone("US/Mountain")
day = 12
month = 8
year = 2015

unixtime = unix_time_convert(year, month, day, local_tz)

date = '%s/%s/%s'%(year, month, day)
clock_time = "18:21:00"
naive = datetime.datetime.strptime("%s %s"%(date,clock_time), "%Y/%m/%d %H:%M:%S")
local_dt = local_tz.localize(naive) #assign the time zone
utc_dt = local_dt.astimezone(pytz.utc) 
#this is correct (6 hour difference)
unixtime_local = calendar.timegm(local_dt.timetuple())
unixtime_utc = calendar.timegm(utc_dt.timetuple()) 
"""
#mktime converts a time tuple in "local" time to seconds since the Epoch
#don't use this, use calendar.timegm
unixtime_current = calendar.timegm(datetime.datetime.utcnow().timetuple())
utc_off = (unixtime_local - unixtime_utc)/60/60 #this is 7 hours off?!
dif = unixtime_current-unixtime_utc
dif2 = unixtime_current-unixtime_local
"""
#great, fixed this part...

delta_t = 67.0
pressure = 101325 / 100
temp = 12
atmos_refract = 0.5667

#what does solar position call?
#import numba?
from numba import autojit
@autojit
def solar_position(unixtime, lat, lon, elev, pressure, temp, delta_t,
                   atmos_refract, sst=False):
	jd = julian_day(unixtime)
	jde = julian_ephemeris_day(jd, delta_t)
	jc = julian_century(jd)
	jce = julian_ephemeris_century(jde)
	jme = julian_ephemeris_millennium(jce)
	L = heliocentric_longitude(jme)
	B = heliocentric_latitude(jme)
	R = heliocentric_radius_vector(jme)
	Theta = geocentric_longitude(L)
	beta = geocentric_latitude(B)
	x0 = mean_elongation(jce)
	x1 = mean_anomaly_sun(jce)
	x2 = mean_anomaly_moon(jce)
	x3 = moon_argument_latitude(jce)
	x4 = moon_ascending_longitude(jce)
	delta_psi = longitude_nutation(jce, x0, x1, x2, x3, x4)
	delta_epsilon = obliquity_nutation(jce, x0, x1, x2, x3, x4)
	epsilon0 = mean_ecliptic_obliquity(jme)
	epsilon = true_ecliptic_obliquity(epsilon0, delta_epsilon)
	delta_tau = aberration_correction(R)
	lamd = apparent_sun_longitude(Theta, delta_psi, delta_tau)
	v0 = mean_sidereal_time(jd, jc)
	v = apparent_sidereal_time(v0, delta_psi, epsilon)
	alpha = geocentric_sun_right_ascension(lamd, epsilon, beta)
	delta = geocentric_sun_declination(lamd, epsilon, beta)
	if sst:
		return(v, alpha, delta) #this is all we need for determining sunset, sunrise, and transit?
	m = sun_mean_longitude(jme)
	eot = equation_of_time(m, alpha, delta_psi, epsilon)
	H = local_hour_angle(v, lon, alpha)
	xi = equatorial_horizontal_parallax(R)
	u = uterm(lat)
	x = xterm(u, lat, elev)
	y = yterm(u, lat, elev)
	delta_alpha = parallax_sun_right_ascension(x, xi, H, delta)
	alpha_prime = topocentric_sun_right_ascension(alpha, delta_alpha)
	delta_prime = topocentric_sun_declination(delta, x, y, xi, delta_alpha, H)
	H_prime = topocentric_local_hour_angle(H, delta_alpha)
	e0 = topocentric_elevation_angle_without_atmosphere(lat, delta_prime, H_prime)
	delta_e = atmospheric_refraction_correction(pressure, temp, e0, atmos_refract)
	e = topocentric_elevation_angle(e0, delta_e)
	theta = topocentric_zenith_angle(e)
	theta0 = topocentric_zenith_angle(e0)
	gamma = topocentric_astronomers_azimuth(H_prime, delta_prime, lat)
	phi = topocentric_azimuth_angle(gamma)

	#things we want
	out = theta, theta0, e, e0, phi, eot
	return(out)
	
#apparent zenith, zenith, apparent elevation, elevation, azimute, eot
out = solar_position(unixtime, lat, lon, elev, pressure, temp, delta_t, atmos_refract)
#solution is correct according to www.esrl.noaa.gov/gmd/grad/solcalc/azel.html
#first thing, lets see if we can get a gridded solution
#ts = 12
ncols = 504
nrows = 472
#dims = (ts, ncols, nrows)
dims = (ncols, nrows)
lats = np.ones(dims)*lat
lons = np.ones(dims)*lon
#unix_utcs = np.arange(0,ts)+unixtime_utc
#unixtime_utcs =np.tensordot(unix_utcs, np.ones((ncols, nrows)), axes = 0)
unix_utcs = np.ones(dims)*unixtime_utc
elevs = np.ones(dims)*elev

timein = calendar.timegm(datetime.datetime.utcnow().timetuple())
zen, app_zen, sel, app_sel, azi, eqt = solar_position(unixtime, lats, lons, elevs, pressure, temp, delta_t, atmos_refract)
timeout = calendar.timegm(datetime.datetime.utcnow().timetuple())
print('time is %s seconds' %(timeout-timein))
#perhaps time can be a constant rather than an array. 
#took some time with the large dimensions
#roughly 2 minutes. But not bad for a single day. 

#new, using numba autojit, may have sped up calculations to 1 second per day. Meaning 6.75 hours!!!
#then we want to calculate transit, sunrise, and sunset
#dates = unix_time_convert(time)

@autojit
def solar_transit_sunrise_sunset(unixtime, lat, lon, delta_t, is_grid = True):
	'''
	code modification 08.17.2015
	if is_grid:
		utday = (unixtime // 86400) * 86400
	else:
		utday = np.array([(unixtime // 86400) * 86400])
	'''
	utday = (unixtime // 86400) * 86400
	ttday0 = utday - delta_t
	ttdayn1 = ttday0 - 86400
	ttdayp1 = ttday0 + 86400

	# index 0 is v, 1 is alpha, 2 is delta
	# going to solve where the elevation angle is equal to -0.8333 for sunrise and sunset
	utday_res = solar_position(utday, 0, 0, 0, 0, 0, delta_t, 0, sst=True)
	v = utday_res[0]

	#remember solar_position takes as arguements 
	# solar_position(unixtime, lat, lon, elev, pressure, temp, delta_t, atmos_refract, numthreads=8, sst=False):

	ttday0_res = solar_position(ttday0, 0, 0, 0, 0, 0, delta_t, 0, sst=True)
	ttdayn1_res = solar_position(ttdayn1, 0, 0, 0, 0, 0, delta_t, 0, sst=True)
	ttdayp1_res = solar_position(ttdayp1, 0, 0, 0, 0, 0, delta_t, 0, sst=True)
	#solve things for the Earth without considering lat and lon or elevation. 
	m0 = (ttday0_res[1] - lon - v) / 360
	cos_arg = ((np.sin(np.radians(-0.8333)) - np.sin(np.radians(lat))
			   * np.sin(np.radians(ttday0_res[2]))) /
			   (np.cos(np.radians(lat)) * np.cos(np.radians(ttday0_res[2]))))
	#cos_arg[np.abs(cos_arg) > 1] = np.nan #maybe here we just define a condition for the single unit to check
	if np.abs(cos_arg) > 1:
		cos_arg = np.nan
	H0 = np.degrees(np.arccos(cos_arg)) % 180  #solar noon hour angle
	'''
	if is_grid: #if it is an array of lats and lons
		m = np.empty((3, np.shape(utday)[0], np.shape(utday)[1]))
	else:
	'''
	#m = np.empty((3, len(utday)))
	m = np.empty((3, 1)) #performing calculations one day at a time currently

	m[0] = m0 % 1
	m[1] = (m[0] - H0 / 360)
	m[2] = (m[0] + H0 / 360)

	# need to account for fractions of day that may be the next or previous
	# day in UTC
	add_a_day = m[2] >= 1
	sub_a_day = m[1] < 0
	m[1] = m[1] % 1
	m[2] = m[2] % 1
	vs = v + 360.985647 * m
	n = m + delta_t / 86400

	a = value_compare(ttday0_res[1] - ttdayn1_res[1])
	#a = value_compare(a)
	#a[abs(a) > 2] = a[abs(a) > 2] % 1
	ap = ttday0_res[2] - ttdayn1_res[2]
	#ap[abs(ap) > 2] = ap[abs(ap) > 2] % 1
	ap = value_compare(ap)
	b = ttdayp1_res[1] - ttday0_res[1]
	b = value_compare(b)
	#b[abs(b) > 2] = b[abs(b) > 2] % 1
	bp = ttdayp1_res[2] - ttday0_res[2]
	bp = value_compare(bp)
	#bp[abs(bp) > 2] = bp[abs(bp) > 2] % 1
	c = b - a
	cp = bp - ap

	alpha_prime = ttday0_res[1] + (n * (a + b + c * n)) / 2
	delta_prime = ttday0_res[2] + (n * (ap + bp + cp * n)) / 2
	Hp = (vs + lon - alpha_prime) % 360
	Hp[Hp >= 180] = Hp[Hp >= 180] - 360

	h = np.degrees(np.arcsin(np.sin(np.radians(lat)) *
							 np.sin(np.radians(delta_prime)) +
							 np.cos(np.radians(lat)) *
							 np.cos(np.radians(delta_prime))
							 * np.cos(np.radians(Hp))))

	T = (m[0] - Hp[0] / 360) * 86400
	R = (m[1] + (h[1] + 0.8333) / (360 * np.cos(np.radians(delta_prime[1])) *
								   np.cos(np.radians(lat)) *
								   np.sin(np.radians(Hp[1])))) * 86400
	S = (m[2] + (h[2] + 0.8333) / (360 * np.cos(np.radians(delta_prime[2])) *
								   np.cos(np.radians(lat)) *
								   np.sin(np.radians(Hp[2])))) * 86400

	S[add_a_day] += 86400
	R[sub_a_day] -= 86400

	transit = T + utday
	sunrise = R + utday
	sunset = S + utday
	#output will be in unixtime
	return(transit, sunrise, sunset)

def value_compare(x):
	if np.abs(x) > 2:
		return(x % 1)
	else:
		return(x)
		
def convert_unixtime_utc_local(unixtime, local_tz):
	timestp =datetime.datetime.utcfromtimestamp(int(unixtime))
	utctime = pytz.utc.localize(timestp)
	localtime = utctime.astimezone(local_tz)
	return(utctime, localtime)
	
timein = calendar.timegm(datetime.datetime.utcnow().timetuple())
ts, sr, ss = solar_transit_sunrise_sunset(unix_utcs, lats, lons, delta_t)
timeout = calendar.timegm(datetime.datetime.utcnow().timetuple())
print('time is %s seconds' %(timeout-timein))

#17 seconds! that's roughly 1.72 hours per year calculated
# if we can reduce this to 10 second...
#solution is correct although the necessity to convert from UTC to local time is needed. 

#currently the program calculates the solar_position for just one day
#we need to be able to call solar_position for an entire year?
