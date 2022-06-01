from logic import *

if __name__ == "__main__":

    heuristics_var = {
        "FIRST": cp_model.CHOOSE_FIRST, # 0
        "LOWEST_MIN": cp_model.CHOOSE_LOWEST_MIN, # 1
        "HIGHEST_MAX": cp_model.CHOOSE_HIGHEST_MAX, # 2
        "MIN_DOMAIN_SIZE": cp_model.CHOOSE_MIN_DOMAIN_SIZE, # 3
        "MAX_DOMAIN_SIZE": cp_model.CHOOSE_MAX_DOMAIN_SIZE # 4
    }
    heuristics_domain = {
        "MIN_VALUE": cp_model.SELECT_MIN_VALUE, # 0
        "MAX_VALUE": cp_model.SELECT_MAX_VALUE, # 1
        "LOWER_HALF": cp_model.SELECT_LOWER_HALF, # 2
        "UPPER_HALF": cp_model.SELECT_UPPER_HALF # 3

    }


    # DEF EXAMPLE
    app = AppLogic("/content/drive/MyDrive/topicos-dataset")
    # CONF REGIONS
    app.regions.SetDeparmento("LIMA")
    app.regions.SetProvincia("LIMA")
    ____selected____ = [False, False, False, False, False, False, True, False, False, False, False, False, \
                        False, False, False, False, False, False, False, False, False, False, False, False, \
                        False, False, False, False, False, False, False, False, True, False, False, False, \
                        False, False, False, True, False, True, True]
    app.regions.SetDistritos(____selected____)
    # CONF PEOPLE
    app.people.SetLocation(percentage=.005 * 0.5, radius=0.025)
    app.people.SetAge("50-59")
    app.people.SetDoses(2)
    app.people.SetInfected()
    # CONF HEALTH CENTERS
    app.hcs.SetCenters()
    app.hcs.AsSample()
    app.hcs.SetCapacity(app.people.total, exceed=0.25)

    appSolver = AppSolver(app)

    print(" "*20, end="")
    for nhd in heuristics_domain.keys():
        print(str(nhd) + " "*(20-len(str(nhd))), end="")
    print()
    for nhv, hv in heuristics_var.items():
        print(str(nhv) + " "*(20-len(str(nhv))), end="")
        for hd in heuristics_domain.values():
            try:
                _ = f"{appSolver.Solve_W_Heuristics(hv, hd):.7f}"
                print(_ + " "* (20-len(_)), end="")
            except:
                print("MAX REACHED"+ " "*4, end="")
        print()    