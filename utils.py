### util functions

import time
import torch
import numpy
import scipy.stats as st
from args import args
import os
import sqlite3


lastDisplay = time.time()
def display(string, end = '\n', force = False):
    global lastDisplay
    if time.time() - lastDisplay > 1 or force:
        lastDisplay = time.time()
        print(string, end=end)

def timeToStr(time):
    hours = int(time) // 3600
    minutes = (int(time) % 3600) // 60
    seconds = int(time) % 60
    return "{:d}h{:02d}m{:02d}s".format(hours, minutes, seconds)

def confInterval(scores):
    try:
        scores = scores.numpy()
    except:
        pass
    if scores.shape[0] == 1:
        low, up = -1., -1.
    elif scores.shape[0] < 30:
        low, up = st.t.interval(0.95, df = scores.shape[0] - 1, loc = scores.mean(), scale = st.sem(scores))
    else:
        low, up = st.norm.interval(0.95, loc = scores.mean(), scale = st.sem(scores))
    return low, up

def createCSV(trainSet, validationSet, testSet):
    if args.csv != "":
        f = open(args.csv, "w")
        text = "epochs, "
        for datasetType in [trainSet, validationSet, testSet]:
            for dataset in datasetType:
                text += dataset["name"] + " loss, " + dataset["name"] + " accuracy, "
        f.write(text + "\n")
        f.close()

def updateCSV(stats, epoch = -1):
    if args.csv != "":
        f = open(args.csv, "a")
        text = ""
        if epoch >= 0:
            text += "\n" + str(epoch) + ", "
        for i in range(stats.shape[0]):
            text += str(stats[i,0].item()) + ", " + str(stats[i,1].item()) + ", "
        f.write(text)
        f.close()



def create_table(filename=args.save_stats):
    conn = sqlite3.connect(filename)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS data (key TEXT PRIMARY KEY, value TEXT)''')
    conn.commit()
    conn.close()

def insert_data(key, value):
    conn = sqlite3.connect(args.save_stats)
    cursor = conn.cursor()
    cursor.execute('''INSERT OR REPLACE INTO data (key, value) VALUES (?, ?)''', (key, value))
    conn.commit()
    conn.close()

def get_data(key):
    conn = sqlite3.connect(args.save_stats)
    cursor = conn.cursor()
    cursor.execute('''SELECT value FROM data WHERE key = ?''', (key,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


print(" utils,", end="")
