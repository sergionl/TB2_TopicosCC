from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import tkinter as tk
from PIL import Image,ImageTk
import random

import logic

class ChecklistBox(tk.Frame):
    def __init__(self, parent, choices, **kwargs):
        tk.Frame.__init__(self, parent, **kwargs)

        self.vars = []
        bg = self.cget("background")
        for choice in choices:
            var = tk.StringVar(value=choice)
            self.vars.append(var)
            cb = tk.Checkbutton(self, var=var, text=choice,
                                onvalue=choice, offvalue="",
                                anchor="w", width=20, background=bg,
                                relief="flat", highlightthickness=0
            )
            cb.pack(side="top", fill="x", anchor="w")


    def getCheckedItems(self):
        values = []
        for var in self.vars:
            value =  var.get()
            if value:
                values.append(value)
        return values

class AppGui:
    def __init__(self):
        self.__start_tk__()
        self.__departamento__()
        self.__examples__()

    def __start_tk__(self):
        self.root = Tk()
        self.root.resizable(width=False, height=False) # Lock window size

        # self.root.attributes('-fullscreen', True)
        w, h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry("%dx%d+0+0" % (w, h))

        self.appLogic = logic.AppLogic("./datasets")
        self.botonDep = Button(text = "REINICIAR",command = self.__destroy_tk_app__)
        self.botonDep.place(x=0,y=0)

    def __destroy_tk_app__(self):
        self.root.destroy()
        self.__start_tk__()
        self.__departamento__()
        self.__examples__()

    def __departamento__(self):
        self.comboDep = ttk.Combobox(values=self.appLogic.regions.GetDepartamentos())
        self.comboDep.place(x=50, y=30)
        
        self.botonDep = Button(text = "ACEPTAR",command = self.__provincia__)
        self.botonDep.place(x=220,y=30)

    def __provincia__(self):
        self.appLogic.regions.SetDeparmento(self.comboDep.get())
        self.comboProv = ttk.Combobox(values=self.appLogic.regions.GetProvincias())
        self.comboProv.place(x=50, y=50)
        
        self.botonProv = Button(text = "ACEPTAR",command = self.__distrito__)
        self.botonProv.place(x=220,y=50)
    
    def __distrito__(self):
        self.appLogic.regions.SetProvincia(self.comboProv.get())

        self.checklistDistritos = ChecklistBox(self.root, self.appLogic.regions.GetDistritos(), bd=0, relief="sunken", background="white")
        self.checklistDistritos.place(x=50, y=70)
        
        self.botonDist = Button(text = "ACEPTAR",command = self.__edades__)
        self.botonDist.place(x=220,y=70)

    def __edades__(self):
        _selected_dist_ = [(e in self.checklistDistritos.getCheckedItems()) for e in self.appLogic.regions.GetDistritos()]
        self.appLogic.regions.SetDistritos(_selected_dist_)
        self.appLogic.people.SetLocation(percentage=.005, radius=0.025)

        self.comboEdades = ttk.Combobox(values=list( self.appLogic.people.__age_group_to_num__.keys() ))
        self.comboEdades.place(x=350, y=30)
        
        self.botonEdades = Button(text = "ACEPTAR",command = self.__num_dosis__)
        self.botonEdades.place(x=520,y=30)

    def __num_dosis__(self):
        self.appLogic.people.SetAge(self.comboEdades.get())

        self.comboDosis = ttk.Combobox(values=list( range(5) ))
        self.comboDosis.place(x=350, y=50)
        
        self.botonDosis = Button(text = "ACEPTAR",command = self.__solve__)
        self.botonDosis.place(x=520,y=50)       

    def __solve__(self):
        self.appLogic.people.SetDoses(int(self.comboDosis.get()))
        self.appLogic.people.SetInfected()
        # Conf Health Centers
        self.appLogic.hcs.SetCenters()
        self.appLogic.hcs.AsSample()
        self.appLogic.hcs.SetCapacity(self.appLogic.people.total, exceed=0.25)
        # Now Solve
        appSolver = logic.AppSolver(self.appLogic)

        appSolver.Solve(False)

        self.appLogic.Draw(appSolver.__plotLines__)

    def __examples__(self):
        self.comboEJEMPLO = ttk.Combobox(values=list( ["EJEMPLO LIMA", "EJEMPLO AREQUIPA", "EJEMPLO ANCASH"] ))
        self.comboEJEMPLO.place(x=350, y=80)

        self.botonEJEMPLO = Button(text = "RESOLVER EJEMPLO",command = self.__solve_example__)
        self.botonEJEMPLO.place(x=520,y=80)      

    def __solve_example__(self):
        to = {"EJEMPLO LIMA": 1, "EJEMPLO AREQUIPA": 2, "EJEMPLO ANCASH": 3}

        if to[self.comboEJEMPLO.get()] == 1:
            self.appLogic.Example1()
        elif to[self.comboEJEMPLO.get()] == 2:
            self.appLogic.Example2()
        elif to[self.comboEJEMPLO.get()] == 3:
            self.appLogic.Example3()

        appSolver = logic.AppSolver(self.appLogic)

        appSolver.Solve(False)

        self.appLogic.Draw(appSolver.__plotLines__)


if __name__ == "__main__":
    
    app = AppGui()

    app.root.mainloop()