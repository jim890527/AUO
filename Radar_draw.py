# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 17:42:05 2021

@author: DGE
"""
import sqlalchemy
import mysql.connector as mysql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DB:
    def connDB():
        db = mysql.connect(
            host='localhost',
            user = 'root',
            password = '',
            port = 8888,
            db = 'badminton'
        )
        return db
    
    def table(table, cursor):    # print database table
        cursor.execute('select * from ' + str(table))
        datas = cursor.fetchall()
        for row in datas:
            print (row)
    
    def creat(data,name):
        connection = sqlalchemy.create_engine("mysql+pymysql://root:@localhost:8888/badminton",encoding="utf-8", echo=False)
        data.to_sql(name=name, con=connection, if_exists='replace')  # append, fail
        return connection
    
    def drop_table(name):   # input table name then del it
        try:
            connection = mysql.connect(host='localhost', user='root', password='', port = 8888, db='badminton')
            cursor = connection.cursor()
            run = ('DELETE FROM ' + str(name))
            cursor.execute(run)
            connection.commit()
            print('Delete successfully')
            connection.close()
            cursor.close()
        except mysql.Error as error:
                 print("Failed {}".format(error))
                 
    def drop_player_radar():
        print('\r\nDELETE player_radar table? (1 is True)')
        name = input()
        if name == '1':
            name = 'player_radar'
            DB.drop_table(name)
        else:
            return
    
    def columes_name(table, cursor):     # get col name
        cursor.execute('select * from ' + str(table))
        cursor.fetchall()
        #num_col = len(cursor.description)
        #print(num_col)
        col_name = [i[0] for i in cursor.description]
        #print(col_names[0])
        return col_name
    
    def check(p_id, mode, table):      # check database.table have this tuple = True
        connection = mysql.connect(host='localhost', user='root', password='', port = 8888, db='badminton')
        cursor = connection.cursor()
        col_name = DB.columes_name(table, cursor)    # get col name
        df_sql = pd.read_sql('SELECT * FROM '+ str(table), con=connection)      # get 'badminton.player_radar' table
        connection.close()
        cursor.close()
        p_id = str(p_id)
        mode = str(mode)
        p = df_sql[df_sql[col_name[0]]==p_id]    # == p_id's target
        #print(p.count()['radar'])
        if p.count()[0] == 0:         # DB non have this ID
            return False
        df_sql = p
        m = df_sql[df_sql[col_name[1]]==mode]   # == p_id && mode's target
        if m.count()[1] == 0:         # have this ID but mode different 
            return False
        else:                               # database have this tuple
            return True
    
    def login(user, password):
        if DB.check(user, password, 'user'):
            print('login')
        else:
            print('failed to login')
            return
    
    def convertToBinaryData(filename):      # translate image format to binary
        # Convert digital data to binary format
        with open(filename, 'rb') as file:
            binaryData = file.read()
        return binaryData
    
    def insertBLOB(p_id, mode, photo):    # image to DB
        if DB.check(p_id, mode, 'player_radar') : return     # Determine if it already exists
        print("Inserting BLOB into player_radar table")
        try:
            connection = mysql.connect(host='localhost', user='root', password='', port = 8888, db='badminton')
            cursor = connection.cursor()
            sql_insert_blob_query = """INSERT INTO player_radar(p_id, mode, radar) VALUES (%s,%s,%s)"""
            #Picture = DB.convertToBinaryData(photo)
            # Convert data into tuple format
            insert_blob_tuple = (p_id, mode, photo)     # input tuple
            result = cursor.execute(sql_insert_blob_query, insert_blob_tuple)
            connection.commit()
            print("Image and file inserted successfully as a BLOB into player_radar table", result)
        except mysql.Error as error:
            print("Failed inserting BLOB data into MySQL table {}".format(error))
        finally:    # must run
            if connection.is_connected():
                connection.close()
                cursor.close()
                print("MySQL connection is closed\r\n")

class Game:
    def __init__(self):     # constructor
        #df = pd.read_csv("game1.csv")  
        df = pd.read_excel("game1.xlsx")
        #df.dropna(axis=1, inplace=True) 
        conn = DB.creat(df,'game')   #create table
        print('\r\n')
        df_sql = pd.read_sql('SELECT * FROM game', con=conn)   #select table to pandas
        del df_sql['index']
        Game.to_eng(df_sql)  # dataframe translate to english
        n = df_sql['number'].unique()   # get number
        for x in n:
            Game.radar('Player' + str(x), df_sql, str(x))    # draw radar from dataframe (condition)
        #db.close()
    
    def to_eng(df):
        df['mode'] = df['隨機模式']
        df['number'] = df['編號'].astype(str) 
        df['target'] = df['燈號位置'].astype(str) 
        df['reaction'] = df['反應時間']
        df['start_to_light'] = df['起點到燈號位置時間']
        df['light_to_start'] = df['燈號回起點位置實測時間']
        df.drop(columns=['紀錄時間','隨機模式','編號','燈號位置','反應時間','起點到燈號位置時間','燈號回起點位置實測時間'], axis=1, inplace=True)
        del df['mode']
        
    def add_radar(target, color, ax, angles, df_r):
        condition = df_r['target'] == target
        df_ar = df_r.loc[condition]      #number = numberr and target = input
        #print(df_ar)
        df_ar = df_ar.groupby(['target']).mean()
        #print(df_ar)
        values = df_ar.loc[target].values.tolist()
        #print(values[1])
        values += values[:1]
        #print(values)
        # Draw the outline of our data.
        ax.plot(angles, values, color=color, linewidth=1, label=target)
        ax.fill(angles, values, color=color, alpha=0.25)
    
    def radar(table, df_r, numberr):   # table = title name
        condition = df_r['number'] == numberr   # number = numberr
        df_r = df_r.loc[condition]
        df_gr = df_r.groupby(['number','target']).mean() 
        print(df_gr)
        count = df_gr.groupby(['number']).size()
        #print(count[0])
        if count[0] != 6:
            print('ERROR!')
            print('\r\n')
            return
        else:
            print('\r\n')
            label = ['reaction','start_to_light','light_to_start']  #draw data col
            num = len(label)
            angles = np.linspace(0, 2 * np.pi, num, endpoint=False).tolist()
            angles += angles[:1]
            label += label[:1]
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            Game.add_radar('1', '#FF2D2D', ax, angles, df_r)
            Game.add_radar('2', '#FF8040', ax, angles, df_r)
            Game.add_radar('3', '#FFD306', ax, angles, df_r)
            Game.add_radar('4', '#00DB00', ax, angles, df_r)
            Game.add_radar('5', '#66B3FF', ax, angles, df_r)
            Game.add_radar('6', '#B15BFF', ax, angles, df_r)
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            #print(np.degrees(angles))
            ax.set_thetagrids(np.degrees(angles), label)
            for label, angle in zip(ax.get_xticklabels(), angles):
                if angle in (0, np.pi):
                    label.set_horizontalalignment('center')
                elif 0 < angle < np.pi:
                    label.set_horizontalalignment('left')
                else:
                    label.set_horizontalalignment('right')
            ax.set_ylim(0, 2.5)
            ax.set_rlabel_position(180 / num)
            ax.tick_params(colors='#000000')
            ax.tick_params(axis='y', labelsize=10)
            #ax.grid(color='#FAFAFA', linewidth=5)
            #ax.spines['polar'].set_color('#333333')
            #ax.set_facecolor('#FAFAFA')
            ax.set_title(str(table),size=15)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plt.savefig(fname=r'game/'+str(table)+'.png', bbox_inches='tight', dpi=150)
            DB.insertBLOB(numberr, "game", r'game/'+str(table)+'.png')  # img to database
    
class Mi:
    def __init__(self):     # constructor
        df = pd.read_excel("mi.xlsx")
        print(df)
        conn = DB.creat(df,'mi')
        df_sql = pd.read_sql('SELECT 編號, 1號燈來回時間, 2號燈來回時間, 3號燈來回時間, 4號燈來回時間, 5號燈來回時間, 6號燈來回時間 FROM mi', con=conn)
        df_r=df_sql.rename(columns={'編號':'number','1號燈來回時間':'P1_round','2號燈來回時間':'P2_round','3號燈來回時間':"P3_round",'4號燈來回時間':'P4_round','5號燈來回時間':'P5_round','6號燈來回時間':'P6_round'})
        df_r_sort=df_r.sort_values(by='number').round(2)
        df_r_sort['number'].astype('int')
        df_r_sort['P1_round'].astype('float')
        df_r_sort['P2_round'].astype('float')
        df_r_sort['P3_round'].astype('float')
        df_r_sort['P4_round'].astype('float')
        df_r_sort['P5_round'].astype('float')
        df_r_sort['P6_round'].astype('float')
        df_r_sort = df_r_sort.groupby(['number']).mean().reset_index()
        print(df_r_sort)
        Mi.mi_pic(df_r_sort)
    
    def mi_pic(result):
        #draw 
        labels = ['P1_round','P2_round','P3_round','P4_round','P5_round','P6_round']
        kinds = list(result.iloc[:, 0])
        result = pd.concat([result, result[['P1_round']]], axis=1)
        centers = np.array(result.iloc[:, 1:])
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        #plot and line
        for i in range(len(kinds)):
            fig = plt.figure()
            ax = fig.add_subplot(111, polar=True) 
            ax.plot(angles, centers[i], linewidth=1, label=kinds[i])
            cg = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'NO6']
            N=len(cg)
            angles=[n/float(N)*2*np.pi for n in range(N)]
            angles+=angles[:1]
            plt.title('Mi_Player' + str(kinds[i]))
            plt.fill(angles,centers[i],facecolor='b',alpha=0.25)
            #plt.legend()
            plt.xticks(angles[:-1],cg)
            plt.yticks([0,1,2,3,4,5],color="gray",size=8)
            plt.ylim(0, 5)
            plt.savefig(fname=r'mi/'+'t_'+str(kinds[i])+'.png', bbox_inches='tight', dpi=150)
            DB.insertBLOB(str(kinds[i]), "mi", r'mi/'+'mi_'+str(kinds[i])+'.png')  # img to database
            plt.show()
            
class T:
    def __init__(self):     # Constructor
        df = pd.read_excel("T.xls")
        conn = DB.creat(df,'t')
        df = pd.read_sql('SELECT 編號,起點到1號,1號到2號,2號到3號 ,3號到1號,1號到起點 FROM t', con=conn)
        df_r = df.rename(columns={'編號':'number','起點到1號':'F-P1','1號到2號':'P1-P2','2號到3號':"P2-P3",'3號到1號':'P3-P1','1號到起點':'P1-F'})
        #del df_r['總計時']
        df_r_sort=df_r.sort_values(by='number').round(2) # 0 = x.xx, 1 = x.xxx, 2 = x.xxxx
        df_r_sort['number'].astype('int')
        df_r_sort['F-P1'].astype('float')
        df_r_sort['P1-P2'].astype('float')
        df_r_sort['P2-P3'].astype('float')
        df_r_sort['P3-P1'].astype('float')
        df_r_sort['P1-F'].astype('float')
        df_r_sort = df_r_sort.groupby(['number']).mean().reset_index()
        print(df_r_sort)
        T.t_pic(df_r_sort)
    
    def t_pic(result):
        labels = ['F-P1','P1-P2','P2-P3','P3-P1','P1-F']
        #result=result.groupby(['number']).mean().reset_index()
        kinds=result.apply(list).iloc[:, 0]
        result = pd.concat([result,result[['F-P1']]], axis=1)
        centers = np.array(result.iloc[:, 1:])
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        #plot and line
        for i in range(len(kinds)):
            fig = plt.figure()
            ax = fig.add_subplot(111, polar=True) 
            ax.plot(angles, centers[i], linewidth=1, label=kinds[i])
            cg=['F-P1','P1-P2','P2-P3','P3-P1','P1-F']
            N=len(cg)
            angles=[n/float(N)*2*np.pi for n in range(N)]
            angles+=angles[:1]
            plt.title('T_Player' + str(kinds[i]))
            plt.fill(angles,centers[i],facecolor='b',alpha=0.25)
            #plt.legend()
            plt.xticks(angles[:-1],cg)
            plt.yticks([0,1,2,3,4,5],color="gray",size=8)
            plt.ylim(0, 5)
            plt.savefig(fname=r't/'+'t_'+str(kinds[i])+'.png', bbox_inches='tight', dpi=150)
            DB.insertBLOB(str(kinds[i]), "t", r't/'+'T_'+str(kinds[i])+'.png')  # img to database
            plt.show()
    
def main():
    Game()
    #Mi()
    #T()
    #DB.drop_player_radar()
    #DB.login('ABC123', '123456')
    
if __name__ == "__main__":
    main()