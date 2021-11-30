{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "409d8492-2b6c-4895-8b7d-4f008f277916",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import math as math\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy import special\n",
    "\n",
    "from mpl_toolkits import mplot3d\n",
    "\n",
    "from mpl_toolkits.mplot3d import Axes3D\n",
    "from matplotlib.colors import LogNorm\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b99fc88d-d3b1-4890-9b36-f20d7452d1b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "fig = plt.figure()\n",
    "ax = Axes3D(fig, azim=-128, elev=43)\n",
    "s = .05\n",
    "X = np.arange(-2, 2.+s, s)\n",
    "Y = np.arange(-1, 3.+s, s)\n",
    "X, Y = np.meshgrid(X, Y)\n",
    "Z = (1.-X)**2 + 100.*(Y-X*X)**2\n",
    "# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, norm = LogNorm(),\n",
    "#                 cmap=\"viridis\")\n",
    "# Without using `` linewidth=0, edgecolor='none' '', the code may produce a\n",
    "# graph with wide black edges, which will make the surface look much darker\n",
    "# than the one illustrated in the figure above.\n",
    "ax.plot_surface(X, Y, Z, rstride=1, cstride=1, norm=LogNorm(),\n",
    "                linewidth=0, edgecolor='none', cmap=\"viridis\")\n",
    "\n",
    "# Set the axis limits so that they are the same as in the figure above.\n",
    "ax.set_xlim([-2, 2.0])                                                       \n",
    "ax.set_ylim([-1, 3.0])                                                       \n",
    "ax.set_zlim([0, 2500]) \n",
    "\n",
    "plt.xlabel(\"x\")\n",
    "plt.ylabel(\"y\")\n",
    "plt.savefig(\"Rosenbrock function.svg\", bbox_inches=\"tight\")\n",
    "\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
