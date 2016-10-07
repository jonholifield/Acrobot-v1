from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm


def drange(x, y, jump):
    while x < y:
        yield float(x)
        x += jump

def testfunc(x,y):
    return x**2 +  y/x
def testfunc2(x,y):
    return x**2 -  y/x

def plot(X,Y,Z, ZR, xmax, ymax, zmax):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #X, Y, Z = axes3d.get_test_data(0.05)


    ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.2, cmap=cm.coolwarm)
    ax.plot_surface(X, Y, ZR, rstride=8, cstride=4, alpha=0.2)
    cset = ax.contour(X, Y, Z, zdir='z', offset=-zmax, cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, zdir='x', offset=-xmax, cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, zdir='y', offset=-ymax, cmap=cm.coolwarm)

    ax.set_xlabel('X')
    ax.set_xlim(-xmax, xmax)
    ax.set_ylabel('Y')
    ax.set_ylim(-ymax, ymax)
    ax.set_zlabel('Z')
    ax.set_zlim(-zmax, zmax)

    plt.show()


if __name__ == "__main__":
    X=[]
    Y=[]
    Z=[]
    ZR=[]

    for x in drange(-10,10, .1):
        XX = []
        YY = []
        ZZ = []
        ZZR = []
        for y in drange(-10,10, .1):
            XX.append(x)
            YY.append(y)
            ZZ.append(testfunc(x,y))
            ZZR.append(testfunc2(x,y))
        X.append(XX)
        Y.append(YY)
        Z.append(ZZ)
        ZR.append(ZZR)
    plot(X,Y,Z, ZR, 10, 10, 100)