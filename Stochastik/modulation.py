
# Authors: Volker Kuehn

"""
==================================================
some routines for transforming integers to bits, binary to Gray mapping
and vice versa
==================================================

.. autosummary::
   :toctree: generated/

   int2bin             -- mapping integer to binary reprentation
   bin2int             -- mapping bit tuple to integer
   bin2gray            -- mapping binary vector to Gray code word
   gray2bin            -- mapping Gray code word to binary vector

"""
import numpy as np

#__all__ = ['bin2int', 'int2bin', 'bin2gray', 'gray2bin', rcosfilter, rrcosfilter]


def int2bin(n,m):
    'From positive integer to list of m binary bits, msb at index 0'
    bits = np.zeros(m,dtype=int)
    if n:
        run = m-1
        while n:
            n,remainder = divmod(n, 2)
            bits[run] = remainder
            run = run - 1

    return bits
 

def bin2int(bits):
    'From binary bits, msb at index 0 to integer'
    i = 0
    for bit in bits:
        i = i * 2 + bit
    return i

def bin2gray(bits):
    'From binary vector to Gray code word'
    return bits[:1] + [i ^ ishift for i, ishift in zip(bits[:-1], bits[1:])]
 
def gray2bin(bits):
    'From Gray code word to binary vector'
    b = [bits[0]]
    for nextb in bits[1:]: 
        b.append(b[-1] ^ nextb)
    return b


def rcosfilter(N, alpha, Ts, Fs):
    """
    Generates a raised cosine (RC) filter (FIR) impulse response.

    Parameters
    ----------
    N : int
        Length of the filter in samples.

    alpha : float
        Roll off factor (Valid values are [0, 1]).

    Ts : float
        Symbol period in seconds.

    Fs : float
        Sampling Rate in Hz.

    Returns
    -------

    h_rc : 1-D ndarray (float)
        Impulse response of the raised cosine filter.

    time_idx : 1-D ndarray (float)
        Array containing the time indices, in seconds, for the impulse response.
    """

    T_delta = 1/float(Fs)
    time_idx = ((np.arange(N)-N/2))*T_delta
    sample_num = np.arange(N)
    h_rc = np.zeros(N, dtype=float)

    for x in sample_num:
        t = (x-N/2)*T_delta
        if t == 0.0:
            h_rc[x] = 1.0
        elif alpha != 0 and t == Ts/(2*alpha):
            h_rc[x] = (np.pi/4)*(np.sin(np.pi*t/Ts)/(np.pi*t/Ts))
        elif alpha != 0 and t == -Ts/(2*alpha):
            h_rc[x] = (np.pi/4)*(np.sin(np.pi*t/Ts)/(np.pi*t/Ts))
        else:
            h_rc[x] = (np.sin(np.pi*t/Ts)/(np.pi*t/Ts))* \
                    (np.cos(np.pi*alpha*t/Ts)/(1-(((2*alpha*t)/Ts)*((2*alpha*t)/Ts))))

    return time_idx, h_rc

def rrcosfilter(N, alpha, Ts, Fs):
    """
    Generates a root raised cosine (RRC) filter (FIR) impulse response.

    Parameters
    ----------
    N : int
        Length of the filter in samples.

    alpha : float
        Roll off factor (Valid values are [0, 1]).

    Ts : float
        Symbol period in seconds.

    Fs : float
        Sampling Rate in Hz.

    Returns
    ---------

    h_rrc : 1-D ndarray of floats
        Impulse response of the root raised cosine filter.

    time_idx : 1-D ndarray of floats
        Array containing the time indices, in seconds, for
        the impulse response.
    """

    T_delta = 1/float(Fs)
    time_idx = ((np.arange(N)-N/2))*T_delta
    sample_num = np.arange(N)
    h_rrc = np.zeros(N, dtype=float)

    for x in sample_num:
        t = (x-N/2)*T_delta
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4*alpha/np.pi)
        elif alpha != 0 and t == Ts/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                    (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        elif alpha != 0 and t == -Ts/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                    (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        else:
            h_rrc[x] = (np.sin(np.pi*t*(1-alpha)/Ts) +  \
                    4*alpha*(t/Ts)*np.cos(np.pi*t*(1+alpha)/Ts))/ \
                    (np.pi*t*(1-(4*alpha*t/Ts)*(4*alpha*t/Ts))/Ts)

    return time_idx, h_rrc

