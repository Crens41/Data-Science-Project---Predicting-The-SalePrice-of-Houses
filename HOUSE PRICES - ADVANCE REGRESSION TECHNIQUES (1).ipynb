{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "07334ebc",
   "metadata": {},
   "source": [
    "# HOUSE PRICES - ADVANCE REGRESSION TECHNIQUES\n",
    "\n",
    "predicting the sale prices of houses with the different features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "0022c4f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importing our libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 193,
   "id": "20d554f8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import the train data set\n",
    "df= pd.read_csv('House_price/train.csv', low_memory=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "725df701",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Id</th>\n",
       "      <th>MSSubClass</th>\n",
       "      <th>MSZoning</th>\n",
       "      <th>LotFrontage</th>\n",
       "      <th>LotArea</th>\n",
       "      <th>Street</th>\n",
       "      <th>Alley</th>\n",
       "      <th>LotShape</th>\n",
       "      <th>LandContour</th>\n",
       "      <th>Utilities</th>\n",
       "      <th>...</th>\n",
       "      <th>PoolArea</th>\n",
       "      <th>PoolQC</th>\n",
       "      <th>Fence</th>\n",
       "      <th>MiscFeature</th>\n",
       "      <th>MiscVal</th>\n",
       "      <th>MoSold</th>\n",
       "      <th>YrSold</th>\n",
       "      <th>SaleType</th>\n",
       "      <th>SaleCondition</th>\n",
       "      <th>SalePrice</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>60</td>\n",
       "      <td>RL</td>\n",
       "      <td>65.0</td>\n",
       "      <td>8450</td>\n",
       "      <td>Pave</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Reg</td>\n",
       "      <td>Lvl</td>\n",
       "      <td>AllPub</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>2008</td>\n",
       "      <td>WD</td>\n",
       "      <td>Normal</td>\n",
       "      <td>208500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>20</td>\n",
       "      <td>RL</td>\n",
       "      <td>80.0</td>\n",
       "      <td>9600</td>\n",
       "      <td>Pave</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Reg</td>\n",
       "      <td>Lvl</td>\n",
       "      <td>AllPub</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "      <td>2007</td>\n",
       "      <td>WD</td>\n",
       "      <td>Normal</td>\n",
       "      <td>181500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>60</td>\n",
       "      <td>RL</td>\n",
       "      <td>68.0</td>\n",
       "      <td>11250</td>\n",
       "      <td>Pave</td>\n",
       "      <td>NaN</td>\n",
       "      <td>IR1</td>\n",
       "      <td>Lvl</td>\n",
       "      <td>AllPub</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>9</td>\n",
       "      <td>2008</td>\n",
       "      <td>WD</td>\n",
       "      <td>Normal</td>\n",
       "      <td>223500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>70</td>\n",
       "      <td>RL</td>\n",
       "      <td>60.0</td>\n",
       "      <td>9550</td>\n",
       "      <td>Pave</td>\n",
       "      <td>NaN</td>\n",
       "      <td>IR1</td>\n",
       "      <td>Lvl</td>\n",
       "      <td>AllPub</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>2006</td>\n",
       "      <td>WD</td>\n",
       "      <td>Abnorml</td>\n",
       "      <td>140000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>60</td>\n",
       "      <td>RL</td>\n",
       "      <td>84.0</td>\n",
       "      <td>14260</td>\n",
       "      <td>Pave</td>\n",
       "      <td>NaN</td>\n",
       "      <td>IR1</td>\n",
       "      <td>Lvl</td>\n",
       "      <td>AllPub</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>2008</td>\n",
       "      <td>WD</td>\n",
       "      <td>Normal</td>\n",
       "      <td>250000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 81 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \\\n",
       "0   1          60       RL         65.0     8450   Pave   NaN      Reg   \n",
       "1   2          20       RL         80.0     9600   Pave   NaN      Reg   \n",
       "2   3          60       RL         68.0    11250   Pave   NaN      IR1   \n",
       "3   4          70       RL         60.0     9550   Pave   NaN      IR1   \n",
       "4   5          60       RL         84.0    14260   Pave   NaN      IR1   \n",
       "\n",
       "  LandContour Utilities  ... PoolArea PoolQC Fence MiscFeature MiscVal MoSold  \\\n",
       "0         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      2   \n",
       "1         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      5   \n",
       "2         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      9   \n",
       "3         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      2   \n",
       "4         Lvl    AllPub  ...        0    NaN   NaN         NaN       0     12   \n",
       "\n",
       "  YrSold  SaleType  SaleCondition  SalePrice  \n",
       "0   2008        WD         Normal     208500  \n",
       "1   2007        WD         Normal     181500  \n",
       "2   2008        WD         Normal     223500  \n",
       "3   2006        WD        Abnorml     140000  \n",
       "4   2008        WD         Normal     250000  \n",
       "\n",
       "[5 rows x 81 columns]"
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "655ad836",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Year House was sold')"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAigAAAHaCAYAAAAqv7IKAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA5LElEQVR4nO3df1yV9f3/8efhN8oPBYUDisQKNcWstCzKQAGV/JW6tNxMN9cqzY0ps6xtslba7Ic6Tatlapnalmk/9GNiGsvMBcxKS80KS4sjyxAECRDf3z+6eb47gSaF8BYe99vtut061/t9XdfrOq/MZ9e5znUcxhgjAAAAi3g1dQEAAADfRUABAADWIaAAAADrEFAAAIB1CCgAAMA6BBQAAGAdAgoAALAOAQUAAFiHgAIAAKxDQAEawJAhQ9SmTRsdPHiw1tjXX3+tqKgoXXPNNTp58mQTVPetCy64QEOGDKlzLC8vTw6HQ8uWLWvcolBvycnJSk5O/t55Bw4coKc4rxFQgAbw1FNPycfHR7/61a9qjd155506duyYli9fLi8v/sgBwNngv5ZAA3A6nVq0aJE2bdqkJ554wr1+7dq1WrVqlR566CFddNFF57SGmpoaVVZWntNjAEBjIaAADWT06NG66aablJmZqQMHDujIkSO6/fbblZaWpjvuuEN5eXkaNmyYwsLCFBAQoMsuu0z/+Mc/PPbx3//+V5MmTVK3bt0UFBSkiIgI9e/fX2+++abHvFOX7+fMmaP7779fcXFx8vf319atWxv0nLZt26aUlBQFBwerVatWSkxM1Pr16z3mZGVlyeFw1Np22bJlcjgcOnDggHvdli1blJycrPDwcAUGBqpTp04aNWqUjh8/7p5TVVWl+++/X127dpW/v7/at2+vX/ziF/rvf/97xlrXr18vh8Oh3Nxc97o1a9bI4XBo8ODBHnMvueQSjRo1yv36scce03XXXaeIiAi1bt1aPXr00Jw5c1RdXe2x3c6dOzVkyBBFRETI399f0dHRGjx4sA4dOnTG2s5mu2+++UYzZsxQXFyc/Pz81KFDB02ePFlHjx49474l6csvv9To0aMVHBys0NBQjRkzRi6X63u3A2zm09QFAM3JY489ppycHP3yl79U+/btVVVVpaefflpbt27VoEGD1KdPHz3++OMKDQ3V6tWrNWbMGB0/flwTJkyQ9O39KpI0c+ZMOZ1OlZWVae3atUpOTtbrr79e696Dv/3tb+rcubMefvhhhYSEKD4+/oz1GWN04sSJWutrampqrcvJyVFaWpouueQSLVmyRP7+/lq0aJGGDh2qVatWacyYMfV6bw4cOKDBgwerb9++evrpp9WmTRt98cUX2rhxo6qqqtSqVSudPHlSw4cP15tvvqnp06crMTFRn332mWbOnKnk5GTl5eUpMDCwzv0nJSXJ19dXmzdv1hVXXCFJ2rx5swIDA5WTk6Pq6mr5+vqqqKhIu3fv1h133OHe9pNPPtHYsWPd4eC9997TAw88oL179+rpp5+WJJWXlystLU1xcXF67LHHFBkZKZfLpa1bt+rYsWOnPe+z2c4YoxtuuEGvv/66ZsyYob59++r999/XzJkz9fbbb+vtt9+Wv79/nfuvqKhQamqqvvzyS82ePVudO3fW+vXr690fwDoGQIPasGGDkWQkmWeffdYYY0zXrl3NZZddZqqrqz3mDhkyxERFRZmampo693XixAlTXV1tUlJSzIgRI9zrCwoKjCRz4YUXmqqqqrOqKzY21l3X6ZalS5e651911VUmIiLCHDt2zKOehIQE07FjR3Py5EljjDEzZ840df2nZOnSpUaSKSgoMMYY88ILLxhJ5t133z1tjatWrTKSzJo1azzW5+bmGklm0aJFZzzHa6+91vTv39/9+qKLLjK///3vjZeXl8nJyTHGGPPcc88ZSeajjz6qcx81NTWmurraPPPMM8bb29t8/fXXxhhj8vLyjCSzbt26M9bwXWez3caNG40kM2fOHI/1zz//vJFknnzySfe6pKQkk5SU5H69ePFiI8m89NJLHtveeuuttXoKnE/4iAdoYOnp6brqqqsUHx+vn//85/r444+1d+9e/exnP5MknThxwr1cf/31Kiws1L59+9zbP/7447r88ssVEBAgHx8f+fr66vXXX9eePXtqHWvYsGHy9fU969quvfZa5ebm1lqeeeYZj3nl5eX697//rZ/+9KcKCgpyr/f29ta4ceN06NAhj5rPxqWXXio/Pz/9+te/1vLly/Xpp5/WmvPqq6+qTZs2Gjp0qMf7dOmll8rpdOqNN9444zFSUlL01ltvqaKiQp999pk+/vhj3XTTTbr00kuVnZ0t6durKp06dfK42rRz504NGzZM4eHh8vb2lq+vr2655RbV1NToo48+kiRddNFFatu2re666y49/vjj+vDDD8/qvM9muy1btkiS+0raKTfeeKNat26t119//bT737p1q4KDgzVs2DCP9WPHjj2r+gBbEVCAc8Df319+fn6SpMOHD0uSMjMz5evr67FMmjRJkvTVV19Jkh599FHdcccd6tOnj9asWaMdO3YoNzdXgwYNUkVFRa3jREVF1auu0NBQ9e7du9Zy8cUXe8wrLi6WMabO/UdHR0uSjhw5Uq9jX3jhhdq8ebMiIiI0efJkXXjhhbrwwgs1f/5895zDhw/r6NGj8vPzq/VeuVwu9/t0OqmpqaqsrNS2bduUnZ2tdu3a6bLLLlNqaqo2b94sSXr99deVmprq3ubzzz9X37599cUXX2j+/Pl68803lZubq8cee0yS3O97aGiocnJydOmll+qee+5R9+7dFR0drZkzZ9a6V+V/nc12R44ckY+Pj9q3b++xrcPhkNPpPON7feTIEUVGRtZa73Q6z/heAbbjHhTgHGvXrp0kacaMGRo5cmSdc7p06SJJWrFihZKTk7V48WKP8dPd41DXzakNoW3btvLy8lJhYWGtsS+//FLS/z+vgIAASVJlZaXHfRJ1hYm+ffuqb9++qqmpUV5enhYsWKCMjAxFRkbqpptuUrt27RQeHq6NGzfWWVdwcPAZ6+7Tp4+CgoK0efNmHThwQCkpKXI4HEpJSdEjjzyi3Nxcff755x4BZd26dSovL9eLL76o2NhY9/p333231v579Oih1atXyxij999/X8uWLdN9992nwMBA3X333aet6/u2Cw8P14kTJ/Tf//7XI6QYY+Ryudz31NQlPDxc77zzTq313CSL8x1XUIBzrEuXLoqPj9d7771X59WL3r17u//idTgctW6GfP/99/X22283as2tW7dWnz599OKLL3pcuTl58qRWrFihjh07qnPnzpK+fQDcqTr/1yuvvHLa/Xt7e6tPnz7uqxT/+c9/JH37wLsjR46opqamzvfpVJA7HV9fX1133XXKzs7Wli1blJaWJunbYOTj46M//OEP7sByyqmQ97/vuzFGf//73097HIfDoZ49e2ru3Llq06aNu/7vc7rtTtWzYsUKj/lr1qxReXm5R73f1a9fPx07dkwvv/yyx/qVK1eeVU2ArbiCAjSCJ554Qunp6Ro4cKAmTJigDh066Ouvv9aePXv0n//8R//85z8lffsX9F/+8hfNnDlTSUlJ2rdvn+677z7FxcXV+e2bc2n27NlKS0tTv379lJmZKT8/Py1atEi7d+/WqlWr3H+xX3/99QoLC9PEiRN13333ycfHR8uWLav1VN3HH39cW7Zs0eDBg9WpUyd988037m/InLqicdNNN+m5557T9ddfr9/+9re68sor5evrq0OHDmnr1q0aPny4RowYcca6U1JSNG3aNI/9BgYGKjExUZs2bdIll1yiiIgI9/y0tDT5+fnp5ptv1vTp0/XNN99o8eLFKi4u9tjvq6++qkWLFumGG27QT37yExlj9OKLL+ro0aPuIFSXs9kuLS1NAwcO1F133aXS0lJdc8017m/xXHbZZRo3btxp93/LLbdo7ty5uuWWW/TAAw8oPj5eGzZs0GuvvXbG9wmwXhPeoAs0W0lJSaZ79+4e69577z0zevRoExERYXx9fY3T6TT9+/c3jz/+uHtOZWWlyczMNB06dDABAQHm8ssvN+vWrTPjx483sbGx7nmnvsXz0EMPnXVNsbGxZvDgwXWOnfqWzHe/8fHmm2+a/v37m9atW5vAwEBz1VVXmVdeeaXW9u+8845JTEw0rVu3Nh06dDAzZ840Tz31lMe3eN5++20zYsQIExsba/z9/U14eLhJSkoyL7/8sse+qqurzcMPP2x69uxpAgICTFBQkOnatau57bbbzP79+7/3PN977z0jycTHx3usf+CBB4wkM3Xq1FrbvPLKK+7jdejQwfz+9783//d//2ckma1btxpjjNm7d6+5+eabzYUXXmgCAwNNaGioufLKK82yZcvOWM/ZbldRUWHuuusuExsba3x9fU1UVJS54447THFxsce8736LxxhjDh06ZEaNGmWCgoJMcHCwGTVqlNm+fTvf4sF5zWGMMU2YjwAAAGrhHhQAAGAdAgoAALAOAQUAAFiHgAIAAKxDQAEAANYhoAAAAOuclw9qO3nypL788ksFBwefs0d9AwCAhmWM0bFjxxQdHS0vrzNfIzkvA8qXX36pmJiYpi4DAAD8AAcPHlTHjh3POKdeAWXx4sVavHixDhw4IEnq3r27/vSnPyk9PV3Stz8Vvnz5co9t+vTpox07drhfV1ZWKjMzU6tWrVJFRYVSUlK0aNGi7y30f5363ZKDBw8qJCSkPqcAAACaSGlpqWJiYr73hz+legaUjh076sEHH9RFF10kSVq+fLmGDx+unTt3qnv37pKkQYMGaenSpe5tTv3k/CkZGRl65ZVXtHr1aoWHh2vatGkaMmSI8vPz5e3tfVZ1nPpYJyQkhIACAMB55mxuz/jRj7oPCwvTQw89pIkTJ2rChAk6evSo1q1bV+fckpIStW/fXs8++6zGjBkj6f9/XLNhwwYNHDjwrI5ZWlqq0NBQlZSUEFAAADhP1Ofv7x/8LZ6amhqtXr1a5eXluvrqq93r33jjDUVERKhz58669dZbVVRU5B7Lz89XdXW1BgwY4F4XHR2thIQEbd++/bTHqqysVGlpqccCAACar3oHlF27dikoKEj+/v66/fbbtXbtWnXr1k2SlJ6erueee05btmzRI488otzcXPXv31+VlZWSJJfLJT8/P7Vt29Zjn5GRkXK5XKc95uzZsxUaGupeuEEWAIDmrd7f4unSpYveffddHT16VGvWrNH48eOVk5Ojbt26uT+2kaSEhAT17t1bsbGxWr9+vUaOHHnafRpjzvh51IwZMzR16lT361M32QAAgOap3gHFz8/PfZNs7969lZubq/nz5+uJJ56oNTcqKkqxsbHav3+/JMnpdKqqqkrFxcUeV1GKioqUmJh42mP6+/vL39+/vqUCAIDz1I9+kqwxxv0RzncdOXJEBw8eVFRUlCSpV69e8vX1VXZ2tntOYWGhdu/efcaAAgAAWpZ6XUG55557lJ6erpiYGB07dkyrV6/WG2+8oY0bN6qsrExZWVkaNWqUoqKidODAAd1zzz1q166dRowYIUkKDQ3VxIkTNW3aNIWHhyssLEyZmZnq0aOHUlNTz8kJAgCA80+9Asrhw4c1btw4FRYWKjQ0VJdccok2btyotLQ0VVRUaNeuXXrmmWd09OhRRUVFqV+/fnr++ec9Hsgyd+5c+fj4aPTo0e4HtS1btuysn4ECAACavx/9HJSmwHNQAAA4/zTKc1AAAADOFQIKAACwDgEFAABYh4ACAACsQ0ABAADWqfeTZFuSC+5e39QlNIgDDw5u6hIAAKgXrqAAAADrEFAAAIB1CCgAAMA6BBQAAGAdAgoAALAOAQUAAFiHgAIAAKxDQAEAANYhoAAAAOvwJFmcF3iqLwC0LFxBAQAA1iGgAAAA6xBQAACAdQgoAADAOgQUAABgHQIKAACwDgEFAABYh4ACAACsQ0ABAADWIaAAAADrEFAAAIB1CCgAAMA6BBQAAGAdAgoAALCOT1MXAOD8c8Hd65u6hB/twIODm7oEAGfAFRQAAGAdAgoAALAOAQUAAFiHgAIAAKxDQAEAANYhoAAAAOsQUAAAgHUIKAAAwDoEFAAAYB0CCgAAsE69HnW/ePFiLV68WAcOHJAkde/eXX/605+Unp4uSTLG6M9//rOefPJJFRcXq0+fPnrsscfUvXt39z4qKyuVmZmpVatWqaKiQikpKVq0aJE6duzYcGcFAC0EPzuA5qpeV1A6duyoBx98UHl5ecrLy1P//v01fPhwffDBB5KkOXPm6NFHH9XChQuVm5srp9OptLQ0HTt2zL2PjIwMrV27VqtXr9a2bdtUVlamIUOGqKampmHPDAAAnLfqFVCGDh2q66+/Xp07d1bnzp31wAMPKCgoSDt27JAxRvPmzdO9996rkSNHKiEhQcuXL9fx48e1cuVKSVJJSYmWLFmiRx55RKmpqbrsssu0YsUK7dq1S5s3bz4nJwgAAM4/P/gelJqaGq1evVrl5eW6+uqrVVBQIJfLpQEDBrjn+Pv7KykpSdu3b5ck5efnq7q62mNOdHS0EhIS3HPqUllZqdLSUo8FAAA0X/UOKLt27VJQUJD8/f11++23a+3aterWrZtcLpckKTIy0mN+ZGSke8zlcsnPz09t27Y97Zy6zJ49W6Ghoe4lJiamvmUDAIDzSL0DSpcuXfTuu+9qx44duuOOOzR+/Hh9+OGH7nGHw+Ex3xhTa913fd+cGTNmqKSkxL0cPHiwvmUDAIDzSL0Dip+fny666CL17t1bs2fPVs+ePTV//nw5nU5JqnUlpKioyH1Vxel0qqqqSsXFxaedUxd/f3+FhIR4LAAAoPn60c9BMcaosrJScXFxcjqdys7Odo9VVVUpJydHiYmJkqRevXrJ19fXY05hYaF2797tngMAAFCv56Dcc889Sk9PV0xMjI4dO6bVq1frjTfe0MaNG+VwOJSRkaFZs2YpPj5e8fHxmjVrllq1aqWxY8dKkkJDQzVx4kRNmzZN4eHhCgsLU2Zmpnr06KHU1NRzcoIAAOD8U6+AcvjwYY0bN06FhYUKDQ3VJZdcoo0bNyotLU2SNH36dFVUVGjSpEnuB7Vt2rRJwcHB7n3MnTtXPj4+Gj16tPtBbcuWLZO3t3fDnhkAADhv1SugLFmy5IzjDodDWVlZysrKOu2cgIAALViwQAsWLKjPoQEAQAvCb/EAAADrEFAAAIB1CCgAAMA6BBQAAGAdAgoAALAOAQUAAFiHgAIAAKxDQAEAANYhoAAAAOsQUAAAgHUIKAAAwDoEFAAAYB0CCgAAsA4BBQAAWIeAAgAArENAAQAA1iGgAAAA6xBQAACAdQgoAADAOgQUAABgHQIKAACwDgEFAABYh4ACAACsQ0ABAADWIaAAAADrEFAAAIB1CCgAAMA6BBQAAGAdAgoAALAOAQUAAFiHgAIAAKxDQAEAANYhoAAAAOsQUAAAgHUIKAAAwDoEFAAAYB0CCgAAsA4BBQAAWIeAAgAArENAAQAA1iGgAAAA69QroMyePVtXXHGFgoODFRERoRtuuEH79u3zmDNhwgQ5HA6P5aqrrvKYU1lZqSlTpqhdu3Zq3bq1hg0bpkOHDv34swEAAM1CvQJKTk6OJk+erB07dig7O1snTpzQgAEDVF5e7jFv0KBBKiwsdC8bNmzwGM/IyNDatWu1evVqbdu2TWVlZRoyZIhqamp+/BkBAIDznk99Jm/cuNHj9dKlSxUREaH8/Hxdd9117vX+/v5yOp117qOkpERLlizRs88+q9TUVEnSihUrFBMTo82bN2vgwIH1PQcAANDM/Kh7UEpKSiRJYWFhHuvfeOMNRUREqHPnzrr11ltVVFTkHsvPz1d1dbUGDBjgXhcdHa2EhARt3769zuNUVlaqtLTUYwEAAM3XDw4oxhhNnTpV1157rRISEtzr09PT9dxzz2nLli165JFHlJubq/79+6uyslKS5HK55Ofnp7Zt23rsLzIyUi6Xq85jzZ49W6Ghoe4lJibmh5YNAADOA/X6iOd/3XnnnXr//fe1bds2j/Vjxoxx/3NCQoJ69+6t2NhYrV+/XiNHjjzt/owxcjgcdY7NmDFDU6dOdb8uLS0lpAAA0Iz9oCsoU6ZM0csvv6ytW7eqY8eOZ5wbFRWl2NhY7d+/X5LkdDpVVVWl4uJij3lFRUWKjIyscx/+/v4KCQnxWAAAQPNVr4BijNGdd96pF198UVu2bFFcXNz3bnPkyBEdPHhQUVFRkqRevXrJ19dX2dnZ7jmFhYXavXu3EhMT61k+AABojur1Ec/kyZO1cuVKvfTSSwoODnbfMxIaGqrAwECVlZUpKytLo0aNUlRUlA4cOKB77rlH7dq104gRI9xzJ06cqGnTpik8PFxhYWHKzMxUjx493N/qAQAALVu9AsrixYslScnJyR7rly5dqgkTJsjb21u7du3SM888o6NHjyoqKkr9+vXT888/r+DgYPf8uXPnysfHR6NHj1ZFRYVSUlK0bNkyeXt7//gzAgAA5716BRRjzBnHAwMD9dprr33vfgICArRgwQItWLCgPocHAAAtBL/FAwAArENAAQAA1iGgAAAA6xBQAACAdQgoAADAOgQUAABgHQIKAACwDgEFAABYh4ACAACsQ0ABAADWIaAAAADrEFAAAIB1CCgAAMA6BBQAAGAdAgoAALAOAQUAAFiHgAIAAKxDQAEAANYhoAAAAOsQUAAAgHUIKAAAwDoEFAAAYB0CCgAAsA4BBQAAWIeAAgAArENAAQAA1iGgAAAA6xBQAACAdQgoAADAOgQUAABgHQIKAACwDgEFAABYh4ACAACsQ0ABAADWIaAAAADrEFAAAIB1CCgAAMA6BBQAAGAdAgoAALAOAQUAAFiHgAIAAKxTr4Aye/ZsXXHFFQoODlZERIRuuOEG7du3z2OOMUZZWVmKjo5WYGCgkpOT9cEHH3jMqays1JQpU9SuXTu1bt1aw4YN06FDh3782QAAgGahXgElJydHkydP1o4dO5Sdna0TJ05owIABKi8vd8+ZM2eOHn30US1cuFC5ublyOp1KS0vTsWPH3HMyMjK0du1arV69Wtu2bVNZWZmGDBmimpqahjszAABw3vKpz+SNGzd6vF66dKkiIiKUn5+v6667TsYYzZs3T/fee69GjhwpSVq+fLkiIyO1cuVK3XbbbSopKdGSJUv07LPPKjU1VZK0YsUKxcTEaPPmzRo4cGADnRoAADhf/ah7UEpKSiRJYWFhkqSCggK5XC4NGDDAPcff319JSUnavn27JCk/P1/V1dUec6Kjo5WQkOCe812VlZUqLS31WAAAQPP1gwOKMUZTp07Vtddeq4SEBEmSy+WSJEVGRnrMjYyMdI+5XC75+fmpbdu2p53zXbNnz1ZoaKh7iYmJ+aFlAwCA88APDih33nmn3n//fa1atarWmMPh8HhtjKm17rvONGfGjBkqKSlxLwcPHvyhZQMAgPPADwooU6ZM0csvv6ytW7eqY8eO7vVOp1OSal0JKSoqcl9VcTqdqqqqUnFx8WnnfJe/v79CQkI8FgAA0HzVK6AYY3TnnXfqxRdf1JYtWxQXF+cxHhcXJ6fTqezsbPe6qqoq5eTkKDExUZLUq1cv+fr6eswpLCzU7t273XMAAEDLVq9v8UyePFkrV67USy+9pODgYPeVktDQUAUGBsrhcCgjI0OzZs1SfHy84uPjNWvWLLVq1Upjx451z504caKmTZum8PBwhYWFKTMzUz169HB/qwcAALRs9QooixcvliQlJyd7rF+6dKkmTJggSZo+fboqKio0adIkFRcXq0+fPtq0aZOCg4Pd8+fOnSsfHx+NHj1aFRUVSklJ0bJly+Tt7f3jzgYAADQL9QooxpjvneNwOJSVlaWsrKzTzgkICNCCBQu0YMGC+hweAAC0EPwWDwAAsA4BBQAAWIeAAgAArENAAQAA1iGgAAAA6xBQAACAdQgoAADAOgQUAABgHQIKAACwDgEFAABYh4ACAACsQ0ABAADWIaAAAADrEFAAAIB1CCgAAMA6BBQAAGAdAgoAALAOAQUAAFiHgAIAAKzj09QFAADQHFxw9/qmLqFBHHhwcFOXIIkrKAAAwEIEFAAAYB0CCgAAsA4BBQAAWIeAAgAArENAAQAA1iGgAAAA6xBQAACAdQgoAADAOgQUAABgHQIKAACwDgEFAABYh4ACAACsQ0ABAADWIaAAAADrEFAAAIB1CCgAAMA6BBQAAGAdAgoAALAOAQUAAFiHgAIAAKxT74Dyr3/9S0OHDlV0dLQcDofWrVvnMT5hwgQ5HA6P5aqrrvKYU1lZqSlTpqhdu3Zq3bq1hg0bpkOHDv2oEwEAAM1HvQNKeXm5evbsqYULF552zqBBg1RYWOheNmzY4DGekZGhtWvXavXq1dq2bZvKyso0ZMgQ1dTU1P8MAABAs+NT3w3S09OVnp5+xjn+/v5yOp11jpWUlGjJkiV69tlnlZqaKklasWKFYmJitHnzZg0cOLC+JQEAgGbmnNyD8sYbbygiIkKdO3fWrbfeqqKiIvdYfn6+qqurNWDAAPe66OhoJSQkaPv27XXur7KyUqWlpR4LAABovho8oKSnp+u5557Tli1b9Mgjjyg3N1f9+/dXZWWlJMnlcsnPz09t27b12C4yMlIul6vOfc6ePVuhoaHuJSYmpqHLBgAAFqn3RzzfZ8yYMe5/TkhIUO/evRUbG6v169dr5MiRp93OGCOHw1Hn2IwZMzR16lT369LSUkIKAADN2Dn/mnFUVJRiY2O1f/9+SZLT6VRVVZWKi4s95hUVFSkyMrLOffj7+yskJMRjAQAAzdc5DyhHjhzRwYMHFRUVJUnq1auXfH19lZ2d7Z5TWFio3bt3KzEx8VyXAwAAzgP1/oinrKxMH3/8sft1QUGB3n33XYWFhSksLExZWVkaNWqUoqKidODAAd1zzz1q166dRowYIUkKDQ3VxIkTNW3aNIWHhyssLEyZmZnq0aOH+1s9AACgZat3QMnLy1O/fv3cr0/dGzJ+/HgtXrxYu3bt0jPPPKOjR48qKipK/fr10/PPP6/g4GD3NnPnzpWPj49Gjx6tiooKpaSkaNmyZfL29m6AUwIAAOe7egeU5ORkGWNOO/7aa6997z4CAgK0YMECLViwoL6HBwAALQC/xQMAAKxDQAEAANYhoAAAAOsQUAAAgHUIKAAAwDoEFAAAYB0CCgAAsA4BBQAAWIeAAgAArENAAQAA1iGgAAAA6xBQAACAdQgoAADAOgQUAABgHQIKAACwDgEFAABYh4ACAACsQ0ABAADWIaAAAADrEFAAAIB1CCgAAMA6BBQAAGAdAgoAALAOAQUAAFiHgAIAAKxDQAEAANYhoAAAAOsQUAAAgHUIKAAAwDoEFAAAYB0CCgAAsA4BBQAAWIeAAgAArENAAQAA1iGgAAAA6xBQAACAdQgoAADAOgQUAABgHQIKAACwDgEFAABYh4ACAACsU++A8q9//UtDhw5VdHS0HA6H1q1b5zFujFFWVpaio6MVGBio5ORkffDBBx5zKisrNWXKFLVr106tW7fWsGHDdOjQoR91IgAAoPmod0ApLy9Xz549tXDhwjrH58yZo0cffVQLFy5Ubm6unE6n0tLSdOzYMfecjIwMrV27VqtXr9a2bdtUVlamIUOGqKam5oefCQAAaDZ86rtBenq60tPT6xwzxmjevHm69957NXLkSEnS8uXLFRkZqZUrV+q2225TSUmJlixZomeffVapqamSpBUrVigmJkabN2/WwIEDf8TpAACA5qBB70EpKCiQy+XSgAED3Ov8/f2VlJSk7du3S5Ly8/NVXV3tMSc6OloJCQnuOd9VWVmp0tJSjwUAADRfDRpQXC6XJCkyMtJjfWRkpHvM5XLJz89Pbdu2Pe2c75o9e7ZCQ0PdS0xMTEOWDQAALHNOvsXjcDg8Xhtjaq37rjPNmTFjhkpKStzLwYMHG6xWAABgnwYNKE6nU5JqXQkpKipyX1VxOp2qqqpScXHxaed8l7+/v0JCQjwWAADQfDVoQImLi5PT6VR2drZ7XVVVlXJycpSYmChJ6tWrl3x9fT3mFBYWavfu3e45AACgZav3t3jKysr08ccfu18XFBTo3XffVVhYmDp16qSMjAzNmjVL8fHxio+P16xZs9SqVSuNHTtWkhQaGqqJEydq2rRpCg8PV1hYmDIzM9WjRw/3t3oAAEDLVu+AkpeXp379+rlfT506VZI0fvx4LVu2TNOnT1dFRYUmTZqk4uJi9enTR5s2bVJwcLB7m7lz58rHx0ejR49WRUWFUlJStGzZMnl7ezfAKQEAgPNdvQNKcnKyjDGnHXc4HMrKylJWVtZp5wQEBGjBggVasGBBfQ8PAABaAH6LBwAAWIeAAgAArENAAQAA1iGgAAAA6xBQAACAdQgoAADAOgQUAABgHQIKAACwDgEFAABYh4ACAACsQ0ABAADWIaAAAADrEFAAAIB1CCgAAMA6BBQAAGAdAgoAALAOAQUAAFiHgAIAAKxDQAEAANYhoAAAAOsQUAAAgHUIKAAAwDoEFAAAYB0CCgAAsA4BBQAAWIeAAgAArENAAQAA1iGgAAAA6xBQAACAdQgoAADAOgQUAABgHQIKAACwDgEFAABYh4ACAACsQ0ABAADWIaAAAADrEFAAAIB1CCgAAMA6BBQAAGAdAgoAALAOAQUAAFinwQNKVlaWHA6Hx+J0Ot3jxhhlZWUpOjpagYGBSk5O1gcffNDQZQAAgPPYObmC0r17dxUWFrqXXbt2ucfmzJmjRx99VAsXLlRubq6cTqfS0tJ07Nixc1EKAAA4D52TgOLj4yOn0+le2rdvL+nbqyfz5s3Tvffeq5EjRyohIUHLly/X8ePHtXLlynNRCgAAOA+dk4Cyf/9+RUdHKy4uTjfddJM+/fRTSVJBQYFcLpcGDBjgnuvv76+kpCRt3779tPurrKxUaWmpxwIAAJqvBg8offr00TPPPKPXXntNf//73+VyuZSYmKgjR47I5XJJkiIjIz22iYyMdI/VZfbs2QoNDXUvMTExDV02AACwSIMHlPT0dI0aNUo9evRQamqq1q9fL0lavny5e47D4fDYxhhTa93/mjFjhkpKStzLwYMHG7psAABgkXP+NePWrVurR48e2r9/v/vbPN+9WlJUVFTrqsr/8vf3V0hIiMcCAACar3MeUCorK7Vnzx5FRUUpLi5OTqdT2dnZ7vGqqirl5OQoMTHxXJcCAADOEz4NvcPMzEwNHTpUnTp1UlFRke6//36VlpZq/PjxcjgcysjI0KxZsxQfH6/4+HjNmjVLrVq10tixYxu6FAAAcJ5q8IBy6NAh3Xzzzfrqq6/Uvn17XXXVVdqxY4diY2MlSdOnT1dFRYUmTZqk4uJi9enTR5s2bVJwcHBDlwIAAM5TDR5QVq9efcZxh8OhrKwsZWVlNfShAQBAM8Fv8QAAAOsQUAAAgHUIKAAAwDoEFAAAYB0CCgAAsA4BBQAAWIeAAgAArENAAQAA1iGgAAAA6xBQAACAdQgoAADAOgQUAABgHQIKAACwDgEFAABYh4ACAACsQ0ABAADWIaAAAADrEFAAAIB1CCgAAMA6BBQAAGAdAgoAALAOAQUAAFiHgAIAAKxDQAEAANYhoAAAAOsQUAAAgHUIKAAAwDoEFAAAYB0CCgAAsA4BBQAAWIeAAgAArENAAQAA1iGgAAAA6xBQAACAdQgoAADAOgQUAABgHQIKAACwDgEFAABYh4ACAACsQ0ABAADWIaAAAADrNGlAWbRokeLi4hQQEKBevXrpzTffbMpyAACAJZosoDz//PPKyMjQvffeq507d6pv375KT0/X559/3lQlAQAASzRZQHn00Uc1ceJE/epXv9LFF1+sefPmKSYmRosXL26qkgAAgCV8muKgVVVVys/P19133+2xfsCAAdq+fXut+ZWVlaqsrHS/LikpkSSVlpae0zpPVh4/p/tvLOf6fWoM9MIuzaEf9MIe9MIu57Ifp/ZtjPneuU0SUL766ivV1NQoMjLSY31kZKRcLlet+bNnz9af//znWutjYmLOWY3NSei8pq4Ap9ALe9ALe9ALuzRGP44dO6bQ0NAzzmmSgHKKw+HweG2MqbVOkmbMmKGpU6e6X588eVJff/21wsPD65x/vigtLVVMTIwOHjyokJCQpi6nRaMX9qAXdqEf9mgOvTDG6NixY4qOjv7euU0SUNq1aydvb+9aV0uKiopqXVWRJH9/f/n7+3usa9OmzbkssVGFhISct/+yNTf0wh70wi70wx7ney++78rJKU1yk6yfn5969eql7Oxsj/XZ2dlKTExsipIAAIBFmuwjnqlTp2rcuHHq3bu3rr76aj355JP6/PPPdfvttzdVSQAAwBJNFlDGjBmjI0eO6L777lNhYaESEhK0YcMGxcbGNlVJjc7f318zZ86s9fEVGh+9sAe9sAv9sEdL64XDnM13fQAAABoRv8UDAACsQ0ABAADWIaAAAADrEFAAAIB1CCgAAMA6BBQAAGCdJv0tnpbo008/1bZt21RYWChvb2/FxcUpLS3tvH5s8fkmPz9fvXr1auoyUIejR4/qn//8pz7//HPFxsbqxhtvPOvHYqNhFBUV6YMPPlCvXr0UEhKiw4cPa/ny5Tp58qQGDx6sHj16NHWJLcr+/fu1fft2uVwuORwORUZGKjExUfHx8U1d2rln0CjKysrMT3/6U+NwOIzD4TBeXl7G6XQab29vExQUZBYuXNjUJbYYDofD/OQnPzEPPPCAOXToUFOX06KNGjXKrFmzxhhjzAcffGDatWtn2rdvb/r06WMiIyON0+k0H374YRNX2XJs3brVtG7d2jgcDhMVFWXee+8907FjRxMfH2+6dOli/P39zWuvvdbUZbYIR48eNcOGDTMOh8O0adPGdO7c2cTHx5s2bdoYLy8vM3z4cFNSUtLUZZ5TfMTTSKZOnarCwkLt3LlTe/bs0YgRI3TLLbeotLRU8+fP1/Tp07Vy5cqmLrPFSElJ0d/+9jddcMEFGjJkiNatW6eampqmLqvFycnJcf8feWZmpgYMGKBDhw5px44dOnjwoAYPHqyMjIymLbIF+cMf/qAJEyaotLRUU6dO1eDBgzV8+HB99NFH2rt3r6ZMmaI///nPTV1mizBlyhQVFBTo7bffVnFxsfbt26ePPvpIxcXF2r59uwoKCjRlypSmLvPcauqE1FK0a9fO5OXluV9//fXXJiAgwJSXlxtjjFm4cKG59NJLm6q8FsXhcJjDhw+b6upq88ILL5jrr7/eeHt7m8jISDN9+nSzd+/epi6xxQgMDDQff/yxMcaYqKgo85///MdjfN++fSY0NLQJKmuZQkJC3P2orq42Pj4+ZufOne7xjz76iH40ktDQULNjx47Tjr/99tvNvhdcQWkkJ06c8LjPJCgoSCdOnFB5ebkkacCAAdq7d29Tldci+fj4aNSoUVq/fr0+++wzTZ48WS+88IK6deum6667rqnLaxEuueQSbdmyRZLkdDr12WefeYx/9tlnCgwMbIrSWiQ/Pz998803kqSqqiqdPHnS/VqSKioq5Ovr21TltTgOh+MHjTUXBJRGcsUVV2j+/Pnu1/Pnz1f79u3Vvn17SVJZWZmCgoKaqrwWpa4/2B06dNAf//hHffLJJ9q0aZNiYmKaoLKW549//KPuvvtuLVu2TL/5zW/0u9/9TkuWLNH27du1dOlSTZw4UePGjWvqMluMa665Rnfffbfeeust/e53v9Pll1+u+++/X+Xl5Tp+/Lj+8pe/qHfv3k1dZoswdOhQ3XrrrcrLy6s1lpeXp9tvv13Dhg1rgsoaUVNfwmkp8vPzTVhYmHE6naZTp07Gz8/PrFq1yj2+cOFCc8sttzRhhS3HqY94YIcXXnjBdOzY0Xh5eblvInc4HCYgIMBkZGSYEydONHWJLcZHH31kLrroIuNwOEz37t3NF198YYYNG2Z8fHyMj4+Pad++vcnPz2/qMluE4uJiM2jQIONwOEzbtm1Nly5dTNeuXU3btm2Nl5eXSU9PN8XFxU1d5jnFrxk3osLCQr366quqrKxU//791a1bt6YuqUXKycnRNddcIx8fvmVvi5qaGuXn56ugoEAnT55UVFSUevXqpeDg4KYurUU6cuSIwsPD3a9ff/11VVRU6Oqrr/ZYj3Nv7969evvtt+VyuSR9+1Ho1Vdfra5duzZxZeceAQUAAFiH/4VsRMYYbd68udZDd6655hqlpKS0iJuebEEv7HG6XiQmJio1NZVeNDL+bJwfiouL9corr+iWW25p6lLOGa6gNJIvvvhCQ4YM0a5du5SQkKDIyEgZY1RUVKTdu3erZ8+eevnll9WhQ4emLrXZoxf2oBd2oR/nj/fee0+XX355s35+EwGlkQwfPlxlZWVasWKFoqKiPMYKCwv185//XMHBwVq3bl3TFNiC0At70Au70A97lJaWnnH8/fffV1JSEgEFP15QUJDeeust9ezZs87xnTt3qm/fviorK2vkyloeemEPemEX+mEPLy+vM36cZoyRw+Fo1gGFe1AaSWBgoL7++uvTjhcXF/NAqkZCL+xBL+xCP+wRHByse++9V3369KlzfP/+/brtttsauarGxYPaGslNN92k8ePH64UXXlBJSYl7fUlJiV544QX94he/0NixY5uwwpaDXtiDXtiFftjj8ssvlyQlJSXVuVxxxRVq7h+AcAWlkTzyyCM6ceKEfvazn+nEiRPy8/OT9O3jpH18fDRx4kQ99NBDTVxly0Av7EEv7EI/7DF27FhVVFScdtzpdGrmzJmNWFHj4x6URlZaWqq8vDwdPnxY0rf/kvXq1cvjd3rQOOiFPeiFXegHbEBAAQAA1uEjnkZUXl6ulStX1vkApJtvvlmtW7du6hJbDHphD3phF/phj5beC66gNJIPP/xQaWlpOn78uJKSkjwegJSTk6PWrVtr06ZN/D5PI6AX9qAXdqEf9qAXBJRG069fPzmdTi1fvtx949kpVVVVmjBhggoLC7V169YmqrDloBf2oBd2oR/2oBcElEbTqlUr5eXlnTbt7t69W1deeaWOHz/eyJW1PPTCHvTCLvTDHvSC56A0mrZt22r//v2nHf/444/Vtm3bRqyo5aIX9qAXdqEf9qAX3CTbaG699VaNHz9ef/jDH5SWlqbIyEg5HA65XC5lZ2dr1qxZysjIaOoyWwR6YQ96YRf6YQ96Icmg0Tz44IMmKirKOBwO4+XlZby8vIzD4TBRUVHmr3/9a1OX16LQC3vQC7vQD3u09F5wD0oTKCgokMvlkvTtA5Di4uKauKKWi17Yg17YhX7Yo6X2goACAACsw02yjaiiokLbtm3Thx9+WGvsm2++0TPPPNMEVbVM9MIe9MIu9MMeLb4XTfsJU8uxb98+Exsb6/4sMSkpyXz55ZfucZfLZby8vJqwwpaDXtiDXtiFftiDXhjDFZRGctddd6lHjx4qKirSvn37FBISomuuuUaff/55U5fW4tALe9ALu9APe9ALcQWlsURERJj333/fY92kSZNMp06dzCeffNIi0rAt6IU96IVd6Ic96IUxPAelkVRUVMjHx/Ptfuyxx+Tl5aWkpCStXLmyiSpreeiFPeiFXeiHPegFD2prNF27dlVeXp4uvvhij/ULFiyQMUbDhg1rospaHnphD3phF/phD3rBt3gazYgRI7Rq1ao6xxYuXKibb75Zhm98Nwp6YQ96YRf6YQ96wXNQAACAhbiCAgAArENAAQAA1iGgAAAA6xBQAACAdQgoAJq95ORkZWRknHHOBRdcoHnz5jVKPQC+HwEFQC3GGKWmpmrgwIG1xhYtWqTQ0NBGe+T2zp07NWTIEEVERCggIEAXXHCBxowZo6+++qpRjg+gaRBQANTicDi0dOlS/fvf/9YTTzzhXl9QUKC77rpL8+fPV6dOnRr0mNXV1bXWFRUVKTU1Ve3atdNrr72mPXv26Omnn1ZUVJSOHz/eoMcHYBcCCoA6xcTEaP78+crMzFRBQYGMMZo4caJSUlJ05ZVX6vrrr1dQUJAiIyM1btw4jysaGzdu1LXXXqs2bdooPDxcQ4YM0SeffOIeP3DggBwOh/7xj38oOTlZAQEBWrFiRa0atm/frtLSUj311FO67LLLFBcXp/79+2vevHkeASknJ0dXXnml/P39FRUVpbvvvlsnTpw47bkVFRVp6NChCgwMVFxcnJ577rkGetcANBQCCoDTGj9+vFJSUvSLX/xCCxcu1O7duzV//nwlJSXp0ksvVV5enjZu3KjDhw9r9OjR7u3Ky8s1depU5ebm6vXXX5eXl5dGjBihkydPeuz/rrvu0m9+8xvt2bOnzo+TnE6nTpw4obVr1572qZlffPGFrr/+el1xxRV67733tHjxYi1ZskT333//ac9rwoQJOnDggLZs2aIXXnhBixYtUlFR0Q98lwCcE03wA4UAziOHDx827du3N15eXubFF180f/zjH82AAQM85hw8eNBIMvv27atzH0VFRUaS2bVrlzHGmIKCAiPJzJs373uPf8899xgfHx8TFhZmBg0aZObMmWNcLpfHeJcuXczJkyfd6x577DETFBRkampqjDHGJCUlmd/+9rfGGGP27dtnJJkdO3a45+/Zs8dIMnPnzj2r9wTAuccVFABnFBERoV//+te6+OKLNWLECOXn52vr1q0KCgpyL127dpUk98c4n3zyicaOHauf/OQnCgkJUVxcnCTVurG2d+/e33v8Bx54QC6XS48//ri6deumxx9/XF27dtWuXbskSXv27NHVV18th8Ph3uaaa65RWVmZDh06VGt/e/bskY+Pj8exu3btqjZt2tTvjQFwTvFrxgC+l4+Pj/un30+ePKmhQ4fqr3/9a615UVFRkqShQ4cqJiZGf//73xUdHa2TJ08qISFBVVVVHvNbt259VscPDw/XjTfeqBtvvFGzZ8/WZZddpocffljLly+XMcYjnEhyfxz03fXfNwbAHgQUAPVy+eWXa82aNbrgggvcoeV/HTlyRHv27NETTzyhvn37SpK2bdvWYMf38/PThRdeqPLycklSt27dtGbNGo+gsn37dgUHB6tDhw61tr/44ot14sQJ5eXl6corr5Qk7du3T0ePHm2wGgH8eHzEA6BeJk+erK+//lo333yz3nnnHX366afatGmTfvnLX6qmpkZt27ZVeHi4nnzySX388cfasmWLpk6d+oOO9eqrr+rnP/+5Xn31VX300Ufat2+fHn74YW3YsEHDhw+XJE2aNEkHDx7UlClTtHfvXr300kuaOXOmpk6dKi+v2v+J69KliwYNGqRbb71V//73v5Wfn69f/epXCgwM/FHvC4CGRUABUC/R0dF66623VFNTo4EDByohIUG//e1vFRoaKi8vL3l5eWn16tXKz89XQkKCfve73+mhhx76Qcfq1q2bWrVqpWnTpunSSy/VVVddpX/84x966qmnNG7cOElShw4dtGHDBr3zzjvq2bOnbr/9dk2cOFF/+MMfTrvfpUuXKiYmRklJSRo5cqR+/etfKyIi4gfVCODccBhzmu/uAQAANBGuoAAAAOsQUAAAgHUIKAAAwDoEFAAAYB0CCgAAsA4BBQAAWIeAAgAArENAAQAA1iGgAAAA6xBQAACAdQgoAADAOv8PskZeXA9+vNkAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Year of the most sold houses\n",
    "df.YrSold.value_counts().plot(kind='bar')\n",
    "plt.xlabel('Year Sold')\n",
    "plt.title('Year House was sold')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "992f90fa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6     253\n",
       "7     234\n",
       "5     204\n",
       "4     141\n",
       "8     122\n",
       "3     106\n",
       "10     89\n",
       "11     79\n",
       "9      63\n",
       "12     59\n",
       "1      58\n",
       "2      52\n",
       "Name: MoSold, dtype: int64"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Months of the most sold houses\n",
    "df.MoSold.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "ae1e438f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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uRmjR49u3WZlZYURI0YgOzsbANCvXz98/fXXuHjxIpYvX47+/fvjueeew4gRI1BfXw8AyM7ObnMMABg9enTrMX4vOzsbERERsLOza90WFxent/MmIsNjASIio/bkk0/ib3/7G/7+97+3u/0lSRKA64+o/37777cFBgZi1qxZeO+993Ds2DGcPXsWO3bsaP15Z47x+88lor6LBYiIjNq9996LpqYmNDU1YcKECW1+NmjQIFhbW+PAgQOt25qbm3H06FGEhobe8Jh+fn6ws7NDXV0dACA0NLTNMQDg0KFDNzxGWFgYTpw40eYJtZ9++qnL50ZE4liKDkBEdDNyubz1VpRcLm/zs379+mHOnDlYvHgxnJyc4OPjg1deeQX19fV46qmnAAAvvfQS6uvrcd9998HX1xfV1dV466230NzcjHvuuQcAsHjxYjz66KOIjo7GXXfdhS+//BI7d+7Ed99912Gm6dOnY9myZXjqqaewfPly5Obm3nC8EBEZJxYgIjJ6Dg4ON/zZunXroNPpMGPGDNTU1CA2NhbffvstHB0dAQDjxo3DO++8g5kzZ+Lq1atwdHREVFQUdu/ejeDgYADAlClT8Oabb+LVV1/Fs88+C39/f7z//vsYP358h5/Zv39/fPnll0hISEBUVBTCwsLw8ssv4+GHH9b7uRORYcgk3swmIiIiM8MxQERERGR2WICIiIjI7LAAERERkdlhASIiIiKzwwJEREREZocFiIiIiMwOCxARERGZHRYgIiIiMjssQERERGR2WICIiIjI7LAAERERkdn5/ypzQG2pW4N6AAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "for col in df['MoSold']:\n",
    "    sns.kdeplot(df['MoSold'], shade=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "d8ea744a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Id</th>\n",
       "      <th>MSSubClass</th>\n",
       "      <th>MSZoning</th>\n",
       "      <th>LotFrontage</th>\n",
       "      <th>LotArea</th>\n",
       "      <th>Street</th>\n",
       "      <th>Alley</th>\n",
       "      <th>LotShape</th>\n",
       "      <th>LandContour</th>\n",
       "      <th>Utilities</th>\n",
       "      <th>...</th>\n",
       "      <th>PoolQC</th>\n",
       "      <th>Fence</th>\n",
       "      <th>MiscFeature</th>\n",
       "      <th>MiscVal</th>\n",
       "      <th>MoSold</th>\n",
       "      <th>YrSold</th>\n",
       "      <th>SaleType</th>\n",
       "      <th>SaleCondition</th>\n",
       "      <th>SalePrice</th>\n",
       "      <th>L_Span</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>60</td>\n",
       "      <td>RL</td>\n",
       "      <td>65.0</td>\n",
       "      <td>8450</td>\n",
       "      <td>Pave</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Reg</td>\n",
       "      <td>Lvl</td>\n",
       "      <td>AllPub</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>2008</td>\n",
       "      <td>WD</td>\n",
       "      <td>Normal</td>\n",
       "      <td>208500</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>20</td>\n",
       "      <td>RL</td>\n",
       "      <td>80.0</td>\n",
       "      <td>9600</td>\n",
       "      <td>Pave</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Reg</td>\n",
       "      <td>Lvl</td>\n",
       "      <td>AllPub</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "      <td>2007</td>\n",
       "      <td>WD</td>\n",
       "      <td>Normal</td>\n",
       "      <td>181500</td>\n",
       "      <td>31</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>60</td>\n",
       "      <td>RL</td>\n",
       "      <td>68.0</td>\n",
       "      <td>11250</td>\n",
       "      <td>Pave</td>\n",
       "      <td>NaN</td>\n",
       "      <td>IR1</td>\n",
       "      <td>Lvl</td>\n",
       "      <td>AllPub</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>9</td>\n",
       "      <td>2008</td>\n",
       "      <td>WD</td>\n",
       "      <td>Normal</td>\n",
       "      <td>223500</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>70</td>\n",
       "      <td>RL</td>\n",
       "      <td>60.0</td>\n",
       "      <td>9550</td>\n",
       "      <td>Pave</td>\n",
       "      <td>NaN</td>\n",
       "      <td>IR1</td>\n",
       "      <td>Lvl</td>\n",
       "      <td>AllPub</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>2006</td>\n",
       "      <td>WD</td>\n",
       "      <td>Abnorml</td>\n",
       "      <td>140000</td>\n",
       "      <td>36</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>60</td>\n",
       "      <td>RL</td>\n",
       "      <td>84.0</td>\n",
       "      <td>14260</td>\n",
       "      <td>Pave</td>\n",
       "      <td>NaN</td>\n",
       "      <td>IR1</td>\n",
       "      <td>Lvl</td>\n",
       "      <td>AllPub</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>2008</td>\n",
       "      <td>WD</td>\n",
       "      <td>Normal</td>\n",
       "      <td>250000</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 82 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \\\n",
       "0   1          60       RL         65.0     8450   Pave   NaN      Reg   \n",
       "1   2          20       RL         80.0     9600   Pave   NaN      Reg   \n",
       "2   3          60       RL         68.0    11250   Pave   NaN      IR1   \n",
       "3   4          70       RL         60.0     9550   Pave   NaN      IR1   \n",
       "4   5          60       RL         84.0    14260   Pave   NaN      IR1   \n",
       "\n",
       "  LandContour Utilities  ... PoolQC Fence MiscFeature MiscVal MoSold YrSold  \\\n",
       "0         Lvl    AllPub  ...    NaN   NaN         NaN       0      2   2008   \n",
       "1         Lvl    AllPub  ...    NaN   NaN         NaN       0      5   2007   \n",
       "2         Lvl    AllPub  ...    NaN   NaN         NaN       0      9   2008   \n",
       "3         Lvl    AllPub  ...    NaN   NaN         NaN       0      2   2006   \n",
       "4         Lvl    AllPub  ...    NaN   NaN         NaN       0     12   2008   \n",
       "\n",
       "  SaleType  SaleCondition  SalePrice  L_Span  \n",
       "0       WD         Normal     208500       5  \n",
       "1       WD         Normal     181500      31  \n",
       "2       WD         Normal     223500       6  \n",
       "3       WD        Abnorml     140000      36  \n",
       "4       WD         Normal     250000       8  \n",
       "\n",
       "[5 rows x 82 columns]"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Checking how old the house has been before sale\n",
    "df['L_Span'] = df['YrSold'] - df['YearRemodAdd']\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "f23e66c4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>MSSubClass</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>YrSold</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2006</th>\n",
       "      <td>59.888535</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2007</th>\n",
       "      <td>54.316109</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2008</th>\n",
       "      <td>59.029605</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2009</th>\n",
       "      <td>54.674556</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2010</th>\n",
       "      <td>56.971429</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        MSSubClass\n",
       "YrSold            \n",
       "2006     59.888535\n",
       "2007     54.316109\n",
       "2008     59.029605\n",
       "2009     54.674556\n",
       "2010     56.971429"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mean_MsSubClass = df.groupby(['YrSold']).agg({'MSSubClass': 'mean'})\n",
    "mean_MsSubClass"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 194,
   "id": "784dc1b0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0       208500\n",
       "1       181500\n",
       "2       223500\n",
       "3       140000\n",
       "4       250000\n",
       "         ...  \n",
       "1455    175000\n",
       "1456    210000\n",
       "1457    266500\n",
       "1458    142125\n",
       "1459    147500\n",
       "Name: SalePrice, Length: 1460, dtype: int64"
      ]
     },
     "execution_count": 194,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['SalePrice']"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "61043032",
   "metadata": {},
   "source": [
    "# Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 230,
   "id": "10ecc4b4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importing the neccessary libraries\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 199,
   "id": "e6c04b86",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Id                 0\n",
       "MSSubClass         0\n",
       "MSZoning           0\n",
       "LotFrontage      259\n",
       "LotArea            0\n",
       "                ... \n",
       "MoSold             0\n",
       "YrSold             0\n",
       "SaleType           0\n",
       "SaleCondition      0\n",
       "SalePrice          0\n",
       "Length: 81, dtype: int64"
      ]
     },
     "execution_count": 199,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Checking for empty cells in the dataframe\n",
    "df.isna().count_values()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "490d42df",
   "metadata": {},
   "source": [
    "# Converting non- numerical values to numerical\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 204,
   "id": "523426a5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Checking for string datatype \n",
    "\n",
    "for label, content in df.items():\n",
    "    if pd.api.types.is_string_dtype(content):\n",
    "        print(label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 205,
   "id": "05218149",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1460 entries, 0 to 1459\n",
      "Data columns (total 81 columns):\n",
      " #   Column         Non-Null Count  Dtype   \n",
      "---  ------         --------------  -----   \n",
      " 0   Id             1460 non-null   int64   \n",
      " 1   MSSubClass     1460 non-null   int64   \n",
      " 2   MSZoning       1460 non-null   category\n",
      " 3   LotFrontage    1201 non-null   float64 \n",
      " 4   LotArea        1460 non-null   int64   \n",
      " 5   Street         1460 non-null   category\n",
      " 6   Alley          91 non-null     category\n",
      " 7   LotShape       1460 non-null   category\n",
      " 8   LandContour    1460 non-null   category\n",
      " 9   Utilities      1460 non-null   category\n",
      " 10  LotConfig      1460 non-null   category\n",
      " 11  LandSlope      1460 non-null   category\n",
      " 12  Neighborhood   1460 non-null   category\n",
      " 13  Condition1     1460 non-null   category\n",
      " 14  Condition2     1460 non-null   category\n",
      " 15  BldgType       1460 non-null   category\n",
      " 16  HouseStyle     1460 non-null   category\n",
      " 17  OverallQual    1460 non-null   int64   \n",
      " 18  OverallCond    1460 non-null   int64   \n",
      " 19  YearBuilt      1460 non-null   int64   \n",
      " 20  YearRemodAdd   1460 non-null   int64   \n",
      " 21  RoofStyle      1460 non-null   category\n",
      " 22  RoofMatl       1460 non-null   category\n",
      " 23  Exterior1st    1460 non-null   category\n",
      " 24  Exterior2nd    1460 non-null   category\n",
      " 25  MasVnrType     1452 non-null   category\n",
      " 26  MasVnrArea     1452 non-null   float64 \n",
      " 27  ExterQual      1460 non-null   category\n",
      " 28  ExterCond      1460 non-null   category\n",
      " 29  Foundation     1460 non-null   category\n",
      " 30  BsmtQual       1423 non-null   category\n",
      " 31  BsmtCond       1423 non-null   category\n",
      " 32  BsmtExposure   1422 non-null   category\n",
      " 33  BsmtFinType1   1423 non-null   category\n",
      " 34  BsmtFinSF1     1460 non-null   int64   \n",
      " 35  BsmtFinType2   1422 non-null   category\n",
      " 36  BsmtFinSF2     1460 non-null   int64   \n",
      " 37  BsmtUnfSF      1460 non-null   int64   \n",
      " 38  TotalBsmtSF    1460 non-null   int64   \n",
      " 39  Heating        1460 non-null   category\n",
      " 40  HeatingQC      1460 non-null   category\n",
      " 41  CentralAir     1460 non-null   category\n",
      " 42  Electrical     1459 non-null   category\n",
      " 43  1stFlrSF       1460 non-null   int64   \n",
      " 44  2ndFlrSF       1460 non-null   int64   \n",
      " 45  LowQualFinSF   1460 non-null   int64   \n",
      " 46  GrLivArea      1460 non-null   int64   \n",
      " 47  BsmtFullBath   1460 non-null   int64   \n",
      " 48  BsmtHalfBath   1460 non-null   int64   \n",
      " 49  FullBath       1460 non-null   int64   \n",
      " 50  HalfBath       1460 non-null   int64   \n",
      " 51  BedroomAbvGr   1460 non-null   int64   \n",
      " 52  KitchenAbvGr   1460 non-null   int64   \n",
      " 53  KitchenQual    1460 non-null   category\n",
      " 54  TotRmsAbvGrd   1460 non-null   int64   \n",
      " 55  Functional     1460 non-null   category\n",
      " 56  Fireplaces     1460 non-null   int64   \n",
      " 57  FireplaceQu    770 non-null    category\n",
      " 58  GarageType     1379 non-null   category\n",
      " 59  GarageYrBlt    1379 non-null   float64 \n",
      " 60  GarageFinish   1379 non-null   category\n",
      " 61  GarageCars     1460 non-null   int64   \n",
      " 62  GarageArea     1460 non-null   int64   \n",
      " 63  GarageQual     1379 non-null   category\n",
      " 64  GarageCond     1379 non-null   category\n",
      " 65  PavedDrive     1460 non-null   category\n",
      " 66  WoodDeckSF     1460 non-null   int64   \n",
      " 67  OpenPorchSF    1460 non-null   int64   \n",
      " 68  EnclosedPorch  1460 non-null   int64   \n",
      " 69  3SsnPorch      1460 non-null   int64   \n",
      " 70  ScreenPorch    1460 non-null   int64   \n",
      " 71  PoolArea       1460 non-null   int64   \n",
      " 72  PoolQC         7 non-null      category\n",
      " 73  Fence          281 non-null    category\n",
      " 74  MiscFeature    54 non-null     category\n",
      " 75  MiscVal        1460 non-null   int64   \n",
      " 76  MoSold         1460 non-null   int64   \n",
      " 77  YrSold         1460 non-null   int64   \n",
      " 78  SaleType       1460 non-null   category\n",
      " 79  SaleCondition  1460 non-null   category\n",
      " 80  SalePrice      1460 non-null   int64   \n",
      "dtypes: category(43), float64(3), int64(35)\n",
      "memory usage: 505.4 KB\n"
     ]
    }
   ],
   "source": [
    "# Changing them to category\n",
    "for label, content in df.items():\n",
    "    if pd.api.types.is_string_dtype(content):\n",
    "        df[label]=content.astype('category').cat.as_ordered()\n",
    "\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 206,
   "id": "ddcf7be3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Alley\n",
      "MasVnrType\n",
      "BsmtQual\n",
      "BsmtCond\n",
      "BsmtExposure\n",
      "BsmtFinType1\n",
      "BsmtFinType2\n",
      "Electrical\n",
      "FireplaceQu\n",
      "GarageType\n",
      "GarageFinish\n",
      "GarageQual\n",
      "GarageCond\n",
      "PoolQC\n",
      "Fence\n",
      "MiscFeature\n"
     ]
    }
   ],
   "source": [
    "# FILL CATEGORICAL VALUES(NON NUMERIC VALUE)\n",
    "\n",
    "# Check first\n",
    "for label, content in df.items():\n",
    "    if not pd.api.types.is_numeric_dtype(content):\n",
    "        if pd.isnull(content).sum():\n",
    "            print(label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 208,
   "id": "6d522b31",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert to numeric\n",
    "\n",
    "for label, content in df.items():\n",
    "    if not pd.api.types.is_numeric_dtype(content):\n",
    "        df[label]=pd.Categorical(content).codes + 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "id": "18d559d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Checking again\n",
    "for label, content in df.items():\n",
    "    if not pd.api.types.is_numeric_dtype(content):\n",
    "        if pd.isnull(content).sum():\n",
    "            print(label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "id": "5a68e84d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "LotFrontage\n",
      "MasVnrArea\n",
      "GarageYrBlt\n"
     ]
    }
   ],
   "source": [
    "# Fill NUmeric Column with missing values\n",
    " \n",
    "# Check first for missing Numeric data\n",
    "for label, content in df.items():\n",
    "    if pd.api.types.is_numeric_dtype(content):\n",
    "        if pd.isnull(content).sum():\n",
    "            print(label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "id": "46e39692",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fill NUmeric Column with missing values\n",
    "for label, content in df.items():\n",
    "    if pd.api.types.is_numeric_dtype(content):\n",
    "        if pd.isnull(content).sum():\n",
    "            df[label]=content.fillna(content.median())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "id": "6c2b314a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check again\n",
    "for label, content in df.items():\n",
    "    if pd.api.types.is_numeric_dtype(content):\n",
    "        if pd.isnull(content).sum():\n",
    "            print(label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 159,
   "id": "950a0ac4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Id               0\n",
       "MSSubClass       0\n",
       "MSZoning         0\n",
       "LotFrontage      0\n",
       "LotArea          0\n",
       "                ..\n",
       "MoSold           0\n",
       "YrSold           0\n",
       "SaleType         0\n",
       "SaleCondition    0\n",
       "SalePrice        0\n",
       "Length: 81, dtype: int64"
      ]
     },
     "execution_count": 159,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "101c2170",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "ca9b280b",
   "metadata": {},
   "source": [
    "# Modelling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 215,
   "id": "f51cd5bf",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Assigning data to x and y\n",
    "\n",
    "x = df.drop('SalePrice', axis=1)\n",
    "y = df['SalePrice']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "id": "d65f7341",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((1460, 80), (1460,))"
      ]
     },
     "execution_count": 136,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.shape, y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "20364e81",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Id</th>\n",
       "      <th>MSSubClass</th>\n",
       "      <th>MSZoning</th>\n",
       "      <th>LotFrontage</th>\n",
       "      <th>LotArea</th>\n",
       "      <th>Street</th>\n",
       "      <th>Alley</th>\n",
       "      <th>LotShape</th>\n",
       "      <th>LandContour</th>\n",
       "      <th>Utilities</th>\n",
       "      <th>...</th>\n",
       "      <th>PoolQC</th>\n",
       "      <th>Fence</th>\n",
       "      <th>MiscFeature</th>\n",
       "      <th>MiscVal</th>\n",
       "      <th>MoSold</th>\n",
       "      <th>YrSold</th>\n",
       "      <th>SaleType</th>\n",
       "      <th>SaleCondition</th>\n",
       "      <th>SalePrice</th>\n",
       "      <th>L_Span</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>60</td>\n",
       "      <td>RL</td>\n",
       "      <td>65.0</td>\n",
       "      <td>8450</td>\n",
       "      <td>Pave</td>\n",
       "      <td>0</td>\n",
       "      <td>Reg</td>\n",
       "      <td>Lvl</td>\n",
       "      <td>AllPub</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>2008</td>\n",
       "      <td>WD</td>\n",
       "      <td>Normal</td>\n",
       "      <td>208500</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>20</td>\n",
       "      <td>RL</td>\n",
       "      <td>80.0</td>\n",
       "      <td>9600</td>\n",
       "      <td>Pave</td>\n",
       "      <td>0</td>\n",
       "      <td>Reg</td>\n",
       "      <td>Lvl</td>\n",
       "      <td>AllPub</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "      <td>2007</td>\n",
       "      <td>WD</td>\n",
       "      <td>Normal</td>\n",
       "      <td>181500</td>\n",
       "      <td>31</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>60</td>\n",
       "      <td>RL</td>\n",
       "      <td>68.0</td>\n",
       "      <td>11250</td>\n",
       "      <td>Pave</td>\n",
       "      <td>0</td>\n",
       "      <td>IR1</td>\n",
       "      <td>Lvl</td>\n",
       "      <td>AllPub</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>9</td>\n",
       "      <td>2008</td>\n",
       "      <td>WD</td>\n",
       "      <td>Normal</td>\n",
       "      <td>223500</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>70</td>\n",
       "      <td>RL</td>\n",
       "      <td>60.0</td>\n",
       "      <td>9550</td>\n",
       "      <td>Pave</td>\n",
       "      <td>0</td>\n",
       "      <td>IR1</td>\n",
       "      <td>Lvl</td>\n",
       "      <td>AllPub</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>2006</td>\n",
       "      <td>WD</td>\n",
       "      <td>Abnorml</td>\n",
       "      <td>140000</td>\n",
       "      <td>36</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>60</td>\n",
       "      <td>RL</td>\n",
       "      <td>84.0</td>\n",
       "      <td>14260</td>\n",
       "      <td>Pave</td>\n",
       "      <td>0</td>\n",
       "      <td>IR1</td>\n",
       "      <td>Lvl</td>\n",
       "      <td>AllPub</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>2008</td>\n",
       "      <td>WD</td>\n",
       "      <td>Normal</td>\n",
       "      <td>250000</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 82 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Id  MSSubClass MSZoning  LotFrontage  LotArea Street  Alley LotShape  \\\n",
       "0   1          60       RL         65.0     8450   Pave      0      Reg   \n",
       "1   2          20       RL         80.0     9600   Pave      0      Reg   \n",
       "2   3          60       RL         68.0    11250   Pave      0      IR1   \n",
       "3   4          70       RL         60.0     9550   Pave      0      IR1   \n",
       "4   5          60       RL         84.0    14260   Pave      0      IR1   \n",
       "\n",
       "  LandContour Utilities  ... PoolQC Fence MiscFeature MiscVal MoSold YrSold  \\\n",
       "0         Lvl    AllPub  ...      0     0           0       0      2   2008   \n",
       "1         Lvl    AllPub  ...      0     0           0       0      5   2007   \n",
       "2         Lvl    AllPub  ...      0     0           0       0      9   2008   \n",
       "3         Lvl    AllPub  ...      0     0           0       0      2   2006   \n",
       "4         Lvl    AllPub  ...      0     0           0       0     12   2008   \n",
       "\n",
       "  SaleType  SaleCondition  SalePrice  L_Span  \n",
       "0       WD         Normal     208500       5  \n",
       "1       WD         Normal     181500      31  \n",
       "2       WD         Normal     223500       6  \n",
       "3       WD        Abnorml     140000      36  \n",
       "4       WD         Normal     250000       8  \n",
       "\n",
       "[5 rows x 82 columns]"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 210,
   "id": "a9eac5d2",
   "metadata": {},
   "outputs": [],
   "source": [
    "for label, content in df.items(): \n",
    "        if pd.api.types.is_string_dtype(content):\n",
    "            df_test[label] = content.astype(\"category\").cat.as_ordered() "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 213,
   "id": "f9b78bd1",
   "metadata": {},
   "outputs": [],
   "source": [
    "for label, content in df.items():\n",
    "      if pd.api.types.is_numeric_dtype(content):\n",
    "            if pd.isnull(content).sum():\n",
    "                df[label]=content.fillna(content.median())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 214,
   "id": "250c07ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "for label, content in df.items():\n",
    "      if not pd.api.types.is_numeric_dtype(content):\n",
    "                df[label]=pd.Categorical(content).codes + 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 216,
   "id": "8c7f9821",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Splitting data\n",
    "model = RandomForestRegressor(n_jobs = -1,\n",
    "                             random_state = 42)\n",
    "\n",
    "x_train,x_test, y_train, y_test=train_test_split(x,\n",
    "                                                y,\n",
    "                                                test_size = 0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 217,
   "id": "e6200d10",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.866448336480517"
      ]
     },
     "execution_count": 217,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Fit model\n",
    "model.fit(x_train,y_train)\n",
    "\n",
    "# Scoring model\n",
    "model.score(x_test,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a670b25a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Checking with confusion metrics\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "89c1f844",
   "metadata": {},
   "source": [
    "# Building our Root-Mean-Squared-Error (RMSE) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 240,
   "id": "46ce34d0",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import mean_squared_error\n",
    "def rmse(y_test,y_preds):\n",
    "    return np.sqrt(mean_squared_error(y_test,y_preds))\n",
    "\n",
    "# create a function to evaluate model on a few different levels\n",
    "def show_scores(model):\n",
    "    y_preds = model.predict(x_train)\n",
    "    scores = {'Training MAE is' : mean_absolute_error(y_train, y_preds),\n",
    "             'Training RMSE is' : rmse(y_train,y_preds),\n",
    "             'Training R^2' : r2_score(y_train, y_preds)}\n",
    "    return scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 241,
   "id": "0286b53f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Wall time: 1.25 s\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "RandomForestRegressor(n_jobs=-1, random_state=42)"
      ]
     },
     "execution_count": 241,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "%%time\n",
    "model.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 242,
   "id": "353b4868",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'Training MAE is': 6722.378690068494,\n",
       " 'Training RMSE is': 11323.038611834401,\n",
       " 'Training R^2': 0.9797737236588803}"
      ]
     },
     "execution_count": 242,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "show_scores(model)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bd8cc506",
   "metadata": {},
   "source": [
    "# Hyperparamter tuning with RandomizedSearchCV"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 247,
   "id": "5c4c79f0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 5 folds for each of 10 candidates, totalling 50 fits\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "RandomizedSearchCV(cv=5,\n",
       "                   estimator=RandomForestRegressor(n_jobs=-1, random_state=42),\n",
       "                   param_distributions={'max_depth': [None, 3, 5, 10],\n",
       "                                        'max_features': [0.5, 1, 'sqrt',\n",
       "                                                         'auto'],\n",
       "                                        'min_samples_leaf': array([ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19]),\n",
       "                                        'min_samples_split': array([ 2,  4,  6,  8, 10, 12, 14, 16, 18]),\n",
       "                                        'n_estimators': array([10, 20, 30, 40, 50, 60, 70, 80, 90])},\n",
       "                   verbose=True)"
      ]
     },
     "execution_count": 247,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import RandomizedSearchCV\n",
    "\n",
    "rs = {'n_estimators' : np.arange(10,100,10),\n",
    "        'max_depth': [None, 3, 5, 10],\n",
    "        'min_samples_split': np.arange(2, 20, 2),\n",
    "        'min_samples_leaf': np.arange(1, 20, 2),\n",
    "        'max_features': [0.5, 1, 'sqrt', 'auto']}\n",
    "\n",
    "rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,\n",
    "                                               random_state=42),\n",
    "                          param_distributions=rs,\n",
    "                          n_iter=10,\n",
    "                          cv=5,\n",
    "                          verbose=True)\n",
    "rs_model.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 248,
   "id": "c994f743",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'Training MAE is': 13202.244577792582,\n",
       " 'Training RMSE is': 23344.609597352188,\n",
       " 'Training R^2': 0.9140267804191177}"
      ]
     },
     "execution_count": 248,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "show_scores(rs_model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 249,
   "id": "b886ecb9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'n_estimators': 40,\n",
       " 'min_samples_split': 18,\n",
       " 'min_samples_leaf': 5,\n",
       " 'max_features': 0.5,\n",
       " 'max_depth': None}"
      ]
     },
     "execution_count": 249,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Checking for best parameters\n",
    "\n",
    "rs_model.best_params_"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8c250295",
   "metadata": {},
   "source": [
    "# Train a model with the best hyperparameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 250,
   "id": "ea7ad74b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Wall time: 438 ms\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "RandomForestRegressor(max_features=0.5, min_samples_leaf=5,\n",
       "                      min_samples_split=18, n_estimators=40, random_state=42)"
      ]
     },
     "execution_count": 250,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "%%time\n",
    "best_model = RandomForestRegressor(n_estimators=40,\n",
    "                                  min_samples_split = 18,\n",
    "                                  min_samples_leaf = 5,\n",
    "                                  max_features = 0.5,\n",
    "                                  max_depth = None,\n",
    "                                  random_state = 42)\n",
    "best_model.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b5b28c4d",
   "metadata": {},
   "source": [
    "### Having all trained models for comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 262,
   "id": "bab589dd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'Training MAE is': 6722.378690068494,\n",
       " 'Training RMSE is': 11323.038611834401,\n",
       " 'Training R^2': 0.9797737236588803}"
      ]
     },
     "execution_count": 262,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "show_scores(model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 263,
   "id": "af3e4ba8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'Training MAE is': 13202.244577792582,\n",
       " 'Training RMSE is': 23344.609597352184,\n",
       " 'Training R^2': 0.9140267804191177}"
      ]
     },
     "execution_count": 263,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "show_scores(rs_model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 251,
   "id": "ae0ecda3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'Training MAE is': 13202.244577792582,\n",
       " 'Training RMSE is': 23344.609597352184,\n",
       " 'Training R^2': 0.9140267804191177}"
      ]
     },
     "execution_count": 251,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "show_scores(best_model)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "253a3608",
   "metadata": {},
   "source": [
    "# Making predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 256,
   "id": "f01e1667",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Id</th>\n",
       "      <th>MSSubClass</th>\n",
       "      <th>MSZoning</th>\n",
       "      <th>LotFrontage</th>\n",
       "      <th>LotArea</th>\n",
       "      <th>Street</th>\n",
       "      <th>Alley</th>\n",
       "      <th>LotShape</th>\n",
       "      <th>LandContour</th>\n",
       "      <th>Utilities</th>\n",
       "      <th>...</th>\n",
       "      <th>ScreenPorch</th>\n",
       "      <th>PoolArea</th>\n",
       "      <th>PoolQC</th>\n",
       "      <th>Fence</th>\n",
       "      <th>MiscFeature</th>\n",
       "      <th>MiscVal</th>\n",
       "      <th>MoSold</th>\n",
       "      <th>YrSold</th>\n",
       "      <th>SaleType</th>\n",
       "      <th>SaleCondition</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1461</td>\n",
       "      <td>20</td>\n",
       "      <td>RH</td>\n",
       "      <td>80.0</td>\n",
       "      <td>11622</td>\n",
       "      <td>Pave</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Reg</td>\n",
       "      <td>Lvl</td>\n",
       "      <td>AllPub</td>\n",
       "      <td>...</td>\n",
       "      <td>120</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>MnPrv</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>6</td>\n",
       "      <td>2010</td>\n",
       "      <td>WD</td>\n",
       "      <td>Normal</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1462</td>\n",
       "      <td>20</td>\n",
       "      <td>RL</td>\n",
       "      <td>81.0</td>\n",
       "      <td>14267</td>\n",
       "      <td>Pave</td>\n",
       "      <td>NaN</td>\n",
       "      <td>IR1</td>\n",
       "      <td>Lvl</td>\n",
       "      <td>AllPub</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Gar2</td>\n",
       "      <td>12500</td>\n",
       "      <td>6</td>\n",
       "      <td>2010</td>\n",
       "      <td>WD</td>\n",
       "      <td>Normal</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1463</td>\n",
       "      <td>60</td>\n",
       "      <td>RL</td>\n",
       "      <td>74.0</td>\n",
       "      <td>13830</td>\n",
       "      <td>Pave</td>\n",
       "      <td>NaN</td>\n",
       "      <td>IR1</td>\n",
       "      <td>Lvl</td>\n",
       "      <td>AllPub</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>MnPrv</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>2010</td>\n",
       "      <td>WD</td>\n",
       "      <td>Normal</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1464</td>\n",
       "      <td>60</td>\n",
       "      <td>RL</td>\n",
       "      <td>78.0</td>\n",
       "      <td>9978</td>\n",
       "      <td>Pave</td>\n",
       "      <td>NaN</td>\n",
       "      <td>IR1</td>\n",
       "      <td>Lvl</td>\n",
       "      <td>AllPub</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>6</td>\n",
       "      <td>2010</td>\n",
       "      <td>WD</td>\n",
       "      <td>Normal</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1465</td>\n",
       "      <td>120</td>\n",
       "      <td>RL</td>\n",
       "      <td>43.0</td>\n",
       "      <td>5005</td>\n",
       "      <td>Pave</td>\n",
       "      <td>NaN</td>\n",
       "      <td>IR1</td>\n",
       "      <td>HLS</td>\n",
       "      <td>AllPub</td>\n",
       "      <td>...</td>\n",
       "      <td>144</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2010</td>\n",
       "      <td>WD</td>\n",
       "      <td>Normal</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 80 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \\\n",
       "0  1461          20       RH         80.0    11622   Pave   NaN      Reg   \n",
       "1  1462          20       RL         81.0    14267   Pave   NaN      IR1   \n",
       "2  1463          60       RL         74.0    13830   Pave   NaN      IR1   \n",
       "3  1464          60       RL         78.0     9978   Pave   NaN      IR1   \n",
       "4  1465         120       RL         43.0     5005   Pave   NaN      IR1   \n",
       "\n",
       "  LandContour Utilities  ... ScreenPorch PoolArea PoolQC  Fence MiscFeature  \\\n",
       "0         Lvl    AllPub  ...         120        0    NaN  MnPrv         NaN   \n",
       "1         Lvl    AllPub  ...           0        0    NaN    NaN        Gar2   \n",
       "2         Lvl    AllPub  ...           0        0    NaN  MnPrv         NaN   \n",
       "3         Lvl    AllPub  ...           0        0    NaN    NaN         NaN   \n",
       "4         HLS    AllPub  ...         144        0    NaN    NaN         NaN   \n",
       "\n",
       "  MiscVal MoSold  YrSold  SaleType  SaleCondition  \n",
       "0       0      6    2010        WD         Normal  \n",
       "1   12500      6    2010        WD         Normal  \n",
       "2       0      3    2010        WD         Normal  \n",
       "3       0      6    2010        WD         Normal  \n",
       "4       0      1    2010        WD         Normal  \n",
       "\n",
       "[5 rows x 80 columns]"
      ]
     },
     "execution_count": 256,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Import the test set\n",
    "df_test = pd.read_csv('House_price/test.csv')\n",
    "df_test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 271,
   "id": "05e1bd48",
   "metadata": {},
   "outputs": [],
   "source": [
    "for label, content in df_test.items():\n",
    "    if pd.api.types.is_string_dtype(content):\n",
    "        df_test[label] = content.astype('category').cat.as_ordered()\n",
    "    \n",
    "    if pd.api.types.is_numeric_dtype(content):\n",
    "        if pd.isnull(content).sum():\n",
    "            df_test[label]=content.fillna(content.median())\n",
    "                \n",
    "    if not pd.api.types.is_numeric_dtype(content):\n",
    "            df_test[label]=pd.Categorical(content).codes + 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 272,
   "id": "2514faed",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\dell\\anaconda3\\lib\\site-packages\\sklearn\\base.py:493: FutureWarning: The feature names should match those that were passed during fit. Starting version 1.2, an error will be raised.\n",
      "Feature names unseen at fit time:\n",
      "- Alley_is_missing\n",
      "- BldgType_is_missing\n",
      "- BsmtCond_is_missing\n",
      "- BsmtExposure_is_missing\n",
      "- BsmtFinType1_is_missing\n",
      "- ...\n",
      "Feature names must be in the same order as they were in fit.\n",
      "\n",
      "  warnings.warn(message, FutureWarning)\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "X has 123 features, but RandomForestRegressor is expecting 80 features as input.",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m~\\AppData\\Local\\Temp\\ipykernel_11760\\1359886214.py\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mtest_preds\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpredict\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdf_test\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\ensemble\\_forest.py\u001b[0m in \u001b[0;36mpredict\u001b[1;34m(self, X)\u001b[0m\n\u001b[0;32m    969\u001b[0m         \u001b[0mcheck_is_fitted\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    970\u001b[0m         \u001b[1;31m# Check data\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 971\u001b[1;33m         \u001b[0mX\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_validate_X_predict\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    972\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    973\u001b[0m         \u001b[1;31m# Assign chunk of trees to jobs\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\ensemble\\_forest.py\u001b[0m in \u001b[0;36m_validate_X_predict\u001b[1;34m(self, X)\u001b[0m\n\u001b[0;32m    577\u001b[0m         Validate X whenever one tries to predict, apply, predict_proba.\"\"\"\n\u001b[0;32m    578\u001b[0m         \u001b[0mcheck_is_fitted\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 579\u001b[1;33m         \u001b[0mX\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_validate_data\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mDTYPE\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0maccept_sparse\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m\"csr\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mreset\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mFalse\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    580\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0missparse\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mand\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mindices\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdtype\u001b[0m \u001b[1;33m!=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mintc\u001b[0m \u001b[1;32mor\u001b[0m \u001b[0mX\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mindptr\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdtype\u001b[0m \u001b[1;33m!=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mintc\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    581\u001b[0m             \u001b[1;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"No support for np.int64 index based sparse matrices\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\base.py\u001b[0m in \u001b[0;36m_validate_data\u001b[1;34m(self, X, y, reset, validate_separately, **check_params)\u001b[0m\n\u001b[0;32m    583\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    584\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0mno_val_X\u001b[0m \u001b[1;32mand\u001b[0m \u001b[0mcheck_params\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"ensure_2d\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;32mTrue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 585\u001b[1;33m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_check_n_features\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mreset\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mreset\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    586\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    587\u001b[0m         \u001b[1;32mreturn\u001b[0m \u001b[0mout\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\base.py\u001b[0m in \u001b[0;36m_check_n_features\u001b[1;34m(self, X, reset)\u001b[0m\n\u001b[0;32m    398\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    399\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mn_features\u001b[0m \u001b[1;33m!=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mn_features_in_\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 400\u001b[1;33m             raise ValueError(\n\u001b[0m\u001b[0;32m    401\u001b[0m                 \u001b[1;34mf\"X has {n_features} features, but {self.__class__.__name__} \"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    402\u001b[0m                 \u001b[1;34mf\"is expecting {self.n_features_in_} features as input.\"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mValueError\u001b[0m: X has 123 features, but RandomForestRegressor is expecting 80 features as input."
     ]
    }
   ],
   "source": [
    "test_preds = model.predict(df_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 273,
   "id": "d10f9ff3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'Alley_is_missing',\n",
       " 'BldgType_is_missing',\n",
       " 'BsmtCond_is_missing',\n",
       " 'BsmtExposure_is_missing',\n",
       " 'BsmtFinType1_is_missing',\n",
       " 'BsmtFinType2_is_missing',\n",
       " 'BsmtQual_is_missing',\n",
       " 'CentralAir_is_missing',\n",
       " 'Condition1_is_missing',\n",
       " 'Condition2_is_missing',\n",
       " 'Electrical_is_missing',\n",
       " 'ExterCond_is_missing',\n",
       " 'ExterQual_is_missing',\n",
       " 'Exterior1st_is_missing',\n",
       " 'Exterior2nd_is_missing',\n",
       " 'Fence_is_missing',\n",
       " 'FireplaceQu_is_missing',\n",
       " 'Foundation_is_missing',\n",
       " 'Functional_is_missing',\n",
       " 'GarageCond_is_missing',\n",
       " 'GarageFinish_is_missing',\n",
       " 'GarageQual_is_missing',\n",
       " 'GarageType_is_missing',\n",
       " 'HeatingQC_is_missing',\n",
       " 'Heating_is_missing',\n",
       " 'HouseStyle_is_missing',\n",
       " 'KitchenQual_is_missing',\n",
       " 'LandContour_is_missing',\n",
       " 'LandSlope_is_missing',\n",
       " 'LotConfig_is_missing',\n",
       " 'LotShape_is_missing',\n",
       " 'MSZoning_is_missing',\n",
       " 'MasVnrType_is_missing',\n",
       " 'MiscFeature_is_missing',\n",
       " 'Neighborhood_is_missing',\n",
       " 'PavedDrive_is_missing',\n",
       " 'PoolQC_is_missing',\n",
       " 'RoofMatl_is_missing',\n",
       " 'RoofStyle_is_missing',\n",
       " 'SaleCondition_is_missing',\n",
       " 'SaleType_is_missing',\n",
       " 'Street_is_missing',\n",
       " 'Utilities_is_missing'}"
      ]
     },
     "execution_count": 273,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Balancing columns\n",
    "set(df_test.columns) - set(x_train.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 274,
   "id": "5e1f92b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Deleting unwanted columns\n",
    "df_test.drop(['Alley_is_missing',\n",
    " 'BldgType_is_missing',\n",
    " 'BsmtCond_is_missing',\n",
    " 'BsmtExposure_is_missing',\n",
    " 'BsmtFinType1_is_missing',\n",
    " 'BsmtFinType2_is_missing',\n",
    " 'BsmtQual_is_missing',\n",
    " 'CentralAir_is_missing',\n",
    " 'Condition1_is_missing',\n",
    " 'Condition2_is_missing',\n",
    " 'Electrical_is_missing',\n",
    " 'ExterCond_is_missing',\n",
    " 'ExterQual_is_missing',\n",
    " 'Exterior1st_is_missing',\n",
    " 'Exterior2nd_is_missing',\n",
    " 'Fence_is_missing',\n",
    " 'FireplaceQu_is_missing',\n",
    " 'Foundation_is_missing',\n",
    " 'Functional_is_missing',\n",
    " 'GarageCond_is_missing',\n",
    " 'GarageFinish_is_missing',\n",
    " 'GarageQual_is_missing',\n",
    " 'GarageType_is_missing',\n",
    " 'HeatingQC_is_missing',\n",
    " 'Heating_is_missing',\n",
    " 'HouseStyle_is_missing',\n",
    " 'KitchenQual_is_missing',\n",
    " 'LandContour_is_missing',\n",
    " 'LandSlope_is_missing',\n",
    " 'LotConfig_is_missing',\n",
    " 'LotShape_is_missing',\n",
    " 'MSZoning_is_missing',\n",
    " 'MasVnrType_is_missing',\n",
    " 'MiscFeature_is_missing',\n",
    " 'Neighborhood_is_missing',\n",
    " 'PavedDrive_is_missing',\n",
    " 'PoolQC_is_missing',\n",
    " 'RoofMatl_is_missing',\n",
    " 'RoofStyle_is_missing',\n",
    " 'SaleCondition_is_missing',\n",
    " 'SaleType_is_missing',\n",
    " 'Street_is_missing',\n",
    " 'Utilities_is_missing'], axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 276,
   "id": "8d0be7ce",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([127382.08, 151981.5 , 180801.32, ..., 155212.36, 113083.5 ,\n",
       "       225210.99])"
      ]
     },
     "execution_count": 276,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_preds = model.predict(df_test)\n",
    "test_preds"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 280,
   "id": "42d32897",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((1459, 80), (1459,))"
      ]
     },
     "execution_count": 280,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_test.shape, y_preds.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 285,
   "id": "113b3772",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>YrSold</th>\n",
       "      <th>2006</th>\n",
       "      <th>2007</th>\n",
       "      <th>2008</th>\n",
       "      <th>2009</th>\n",
       "      <th>2010</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>SalePrice</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>34900</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>35311</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>37900</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>39300</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>40000</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>582933</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>611657</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>625000</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>745000</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>755000</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>663 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "YrSold     2006  2007  2008  2009  2010\n",
       "SalePrice                              \n",
       "34900         0     0     0     1     0\n",
       "35311         1     0     0     0     0\n",
       "37900         0     0     0     1     0\n",
       "39300         0     1     0     0     0\n",
       "40000         0     0     1     0     0\n",
       "...         ...   ...   ...   ...   ...\n",
       "582933        0     0     0     1     0\n",
       "611657        0     0     0     0     1\n",
       "625000        1     0     0     0     0\n",
       "745000        0     1     0     0     0\n",
       "755000        0     1     0     0     0\n",
       "\n",
       "[663 rows x 5 columns]"
      ]
     },
     "execution_count": 285,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Compare target column with SaleCondition\n",
    "pd.crosstab(df.SalePrice, df.YrSold)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a31b7ad8",
   "metadata": {},
   "source": [
    "# Feature Importances"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 290,
   "id": "8acc551d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([4.07662642e-03, 1.59376383e-03, 2.31147220e-03, 5.97895469e-03,\n",
       "       1.86083628e-02, 1.37715092e-05, 1.10246311e-04, 1.56184218e-03,\n",
       "       1.67604850e-03, 0.00000000e+00, 7.17116299e-04, 7.66709513e-04,\n",
       "       8.46023510e-03, 4.33610689e-04, 3.44855113e-04, 5.11902238e-04,\n",
       "       6.75793557e-04, 5.91479344e-01, 4.47535850e-03, 7.66149817e-03,\n",
       "       7.79057819e-03, 8.00322707e-04, 4.88605343e-04, 1.35426301e-03,\n",
       "       1.19440747e-03, 8.48271865e-04, 8.19681170e-03, 3.76311845e-03,\n",
       "       5.93705905e-04, 2.98205761e-04, 2.63087224e-03, 5.72968347e-04,\n",
       "       1.73978916e-03, 7.99081287e-04, 2.17477524e-02, 3.24290179e-04,\n",
       "       4.45476032e-04, 5.28076965e-03, 3.79757653e-02, 9.89625529e-05,\n",
       "       7.65415597e-04, 2.14083680e-03, 1.43401868e-04, 3.40286730e-02,\n",
       "       1.94121888e-02, 1.34179394e-04, 1.04903766e-01, 8.19306167e-04,\n",
       "       1.16318771e-03, 4.21508524e-03, 7.07642839e-04, 1.49283206e-03,\n",
       "       1.17546015e-03, 1.99943247e-03, 5.80460139e-03, 6.17949062e-04,\n",
       "       2.75370866e-03, 2.87673220e-03, 3.54439818e-03, 4.25379103e-03,\n",
       "       1.55415834e-03, 2.27829192e-02, 1.34361959e-02, 3.35520698e-04,\n",
       "       5.18590492e-04, 5.37134562e-04, 4.32263913e-03, 5.36345880e-03,\n",
       "       8.95555503e-04, 5.19802164e-05, 1.61999995e-03, 1.53374025e-04,\n",
       "       4.18845595e-05, 3.46260308e-04, 2.17960063e-05, 6.91638197e-05,\n",
       "       3.49559439e-03, 1.54047980e-03, 6.06500420e-04, 9.52675654e-04])"
      ]
     },
     "execution_count": 290,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.feature_importances_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 291,
   "id": "87646fc4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Helper function for plotting feature importance\n",
    "def plot_features(columns, importances, n=20):\n",
    "    df = (pd.DataFrame({'features': columns,\n",
    "                       'feature_importances': importances})\n",
    "         .sort_values('feature_importances', ascending=False)\n",
    "         .reset_index(drop=True))\n",
    "    \n",
    "    #Plot the dataframe\n",
    "    fig, ax=plt.subplots()\n",
    "    ax.barh(df['features'][:n], df['feature_importances'][:20])\n",
    "    ax.set_ylabel('Features')\n",
    "    ax.set_xlabel('Feature Importance')\n",
    "    ax.invert_yaxis()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 293,
   "id": "ce7271d5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_features(x_train.columns, model.feature_importances_)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fe748ce5",
   "metadata": {},
   "source": [
    "# Saving our predicted SalePrice"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 294,
   "id": "07956484",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([127382.08, 151981.5 , 180801.32, ..., 155212.36, 113083.5 ,\n",
       "       225210.99])"
      ]
     },
     "execution_count": 294,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_preds"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 295,
   "id": "d0332b55",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Id</th>\n",
       "      <th>SalePrice</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1461</td>\n",
       "      <td>127382.08</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1462</td>\n",
       "      <td>151981.50</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1463</td>\n",
       "      <td>180801.32</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1464</td>\n",
       "      <td>186866.32</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1465</td>\n",
       "      <td>197088.12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1454</th>\n",
       "      <td>2915</td>\n",
       "      <td>86669.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1455</th>\n",
       "      <td>2916</td>\n",
       "      <td>88198.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1456</th>\n",
       "      <td>2917</td>\n",
       "      <td>155212.36</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1457</th>\n",
       "      <td>2918</td>\n",
       "      <td>113083.50</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1458</th>\n",
       "      <td>2919</td>\n",
       "      <td>225210.99</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1459 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        Id  SalePrice\n",
       "0     1461  127382.08\n",
       "1     1462  151981.50\n",
       "2     1463  180801.32\n",
       "3     1464  186866.32\n",
       "4     1465  197088.12\n",
       "...    ...        ...\n",
       "1454  2915   86669.00\n",
       "1455  2916   88198.00\n",
       "1456  2917  155212.36\n",
       "1457  2918  113083.50\n",
       "1458  2919  225210.99\n",
       "\n",
       "[1459 rows x 2 columns]"
      ]
     },
     "execution_count": 295,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_preds= pd.DataFrame()\n",
    "df_preds['Id'] = df_test['Id']\n",
    "df_preds['SalePrice'] = test_preds\n",
    "df_preds"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 300,
   "id": "517d9911",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Exporting our predicted SalePrice\n",
    "df_preds.to_csv('House_price/House_SalePrice_Submission', index= False)"
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
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
