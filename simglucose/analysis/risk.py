import numpy as np
import warnings


def risk_index(BG, horizon):
    # BG is in mg/dL
    # horizon in samples
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        BG_to_compute = BG[-horizon:]
        #print(BG_to_compute)
        fBG = 1.509 * (np.log(BG_to_compute)**1.084 - 5.381)
        rl = 10 * fBG[fBG < 0]**2
        rh = 10 * fBG[fBG > 0]**2
        LBGI = np.nan_to_num(np.mean(rl))
        HBGI = np.nan_to_num(np.mean(rh))
        RI = LBGI + HBGI
    return (LBGI, HBGI, RI)

def risk_index2(BG, horizon):
    # BG is in mg/dL
    # horizon in samples
    hyper = 180
    hypo = 70
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # TODO: currently supports only horizon 1
        BG_to_compute = BG[-horizon:]
        fBG = BG_to_compute
        # rl = np.abs(fBG[np.array(fBG[0]) < hypo] - hypo) * 2
        rl = np.abs(fBG[0] - hypo) * 5 if fBG[0] < hypo else 0
        # rh = np.abs(fBG[np.array(fBG[0]) > hyper] - hyper)
        rh = np.abs(fBG[0] - hyper) if fBG[0] > hyper else 0
        LBGI = np.nan_to_num(np.mean(rl))
        HBGI = np.nan_to_num(np.mean(rh))
        RI = LBGI + HBGI
    return (LBGI, HBGI, RI)