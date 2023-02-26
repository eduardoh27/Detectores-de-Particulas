import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def main():

    ruta = r"C:\Users\eduar\OneDrive - Universidad de los Andes\OTROS\Sofi-Edu\2023-10\Detectores de partículas\Tarea 2\DataTarea2.xlsx"
    df = pd.read_excel(ruta)

    y = df['dE/dx(MeV/mm)'].tolist()
    x = df['Energy(MeV)'].tolist()

        
    


    #popt = scipyExponential1(y, x)

    #calculoNumerico(popt)
    scipyExponential2(y, x)


def calculoNumerico(popt):

    deltaX = 1
    
    Xanterior = 0 # X inicial
    Eanterior = 100 # E inicial
    i = 0

    while(Enueva > 0):
        i+=1
        print(f"\niteración = {i}")

        Xnueva = Xanterior + deltaX

        Enueva = Eanterior - func1(Eanterior, *popt)*Xnueva
        print(f"Enueva = {Enueva}")

        Xanterior = Xnueva
        Eanterior = Enueva 
        
    
    print(f"El alcance R del protón de 100 MeV en un cristal de Silicio = {Xnueva} mm")


#def scipyExponential1(ydata, xdata):
    
    """
    xdata = np.linspace(0, 4, 50)

    y = func(xdata, 2.5, 1.3, 0.5)

    rng = np.random.default_rng()

    y_noise = 0.2 * rng.normal(size=xdata.size)

    ydata = y + y_noise

    plt.plot(xdata, ydata, 'b-', label='data')
    """

    popt, pcov = curve_fit(func1, xdata, ydata)


    plt.plot(xdata, func1(xdata, *popt), 'r-')
    plt.scatter(xdata, ydata)
    plt.show()

    return popt

def func1(x, a, b, c):
    x = np.array(x)
    return a * np.exp(-b * x) + c


def scipyExponential2(ydata, xdata):
    

    popt, pcov = curve_fit(func2, xdata, ydata)

    xdata.sort()
    print(xdata)

    #xdata = xdata[1:-2]

    plt.plot(xdata, func2(xdata, *popt), 'r-')
    plt.scatter(xdata, ydata)
    plt.show()

def func2(x, a, b):
    x = np.array(x)
    return a * np.exp(-b * x)

def numpyPoly(y, x, nDegree: int):

    mymodel = np.poly1d(np.polyfit(x, y, nDegree))
    myline = np.linspace(1, 100, 10000)

    plt.scatter(x, y)
    plt.plot(myline, mymodel(myline))
    plt.show() 

    print(y)
    print(x)

def example():

    x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
    y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

    mymodel = np.poly1d(np.polyfit(x, y, 3))

    myline = np.linspace(1, 22, 100)

    plt.scatter(x, y)
    plt.plot(myline, mymodel(myline))
    plt.show() 

if __name__ == "__main__":
    main()
