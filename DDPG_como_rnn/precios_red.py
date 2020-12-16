import numpy as np 
import pandas as pd
from datetime import datetime
import time
import matplotlib.pyplot as plt 


class Precios_red():
    def __init__(self):
        filename = 'export_PrecioDeLaEnergíaExcedentariaDelAutoconsumoParaElMecanismoDeCompensaciónSimplificada (PVPC)_2020-12-07_22_18.csv'

        df = pd.read_csv(filename, sep=';')
        # view head
        # df2 = df[df['id']==1739]
        # df2 = df[df['id']==1013]
        # df2 = df[df['id']==1014]
        df2 = df[df['id']==1015]

        datetime_v = df2['datetime'].values
        datetime = np.array([self.convert_datetime_str(date_str) for date_str in datetime_v])
        datetime = datetime - datetime[0]
        self.datetime_hour = datetime / (60*60)
        self.value    = np.array(df2['value'].values)

    def convert_datetime_str(self, date_time_str):
        str_split = date_time_str.split(sep='T')
        str_0 = str_split[0]
        str_0_split = str_0.split('-')
        year  = int(str_0_split[0])
        month = int(str_0_split[1])
        day   = int(str_0_split[2])
        str_1 = str_split[1]
        str_1_split = str_1.split(':')
        hour = int(str_1_split[0])

        a = time.struct_time((year, month, day, hour, 0, 0,0,0,0,'Romance Standard Time',3600))
        x = time.mktime(a)
        return x


if __name__ == "__main__":
    precios = Precios_red()

    Num_dias = 110
    x=[]
    for n in range(Num_dias):
        coste_dia =  precios.value[n*24:(n+1)*24]
        x.append(coste_dia)
    plt.figure()
    for n in range(Num_dias):
        plt.plot(x[n],'.-')
    plt.show(block=True)

    plt.figure()
    plt.plot(precios.datetime_hour, precios.value,'.-', color='black')
    plt.xlabel('hour')
    plt.ylabel('precio')
    plt.show(block=True)


