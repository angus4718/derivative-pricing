from MonteCarlo import MonteCarlo
from Div_Rf import Div_Rf

def main():
    #Configurations
    T = 3
    dt = 1 / 252
    n_sim = 1000

    div_rf = Div_Rf(T, dt)
    r = div_rf.get_all_rf()
    d = div_rf.get_all_dividend()
    r_forward = [div_rf.get_forward_rates(index, dt) for index in r]
    d_forward = [div_rf.get_forward_rates(index, dt) for index in d]
    mc = MonteCarlo(r, r_forward, d_forward, T, dt, n_sim)
    print(mc.bisection(9800))


if __name__ == "__main__":
    main()
