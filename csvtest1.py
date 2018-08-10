import csv

with open('example1.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        print(row)
        
        print(row[0])
        print(row[0],row[1],row[2],)
        print('-'*20)
