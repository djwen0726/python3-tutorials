import csv

with open('example1.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    dates = []
    colors = []
    for row in readCSV:
        color = row[3]
        date = row[0]

        dates.append(date)
        colors.append(color)

    print(dates)
    print(colors)

    try:
        whatColor = input('What color do you wish to know the date of?:')

        if whatColor in colors:
            coldex = colors.index(whatColor)
            theDate = dates[coldex]
            print('The date of',whatColor,'is:',theDate)
        else:
            # now we can handle a specific scenario, instead
            # of handling it with a "catch-all" error.
            # now we have error handling and
            # proper logic. Yay.
            print('This color was not found.')

    # in python 2, this is read exception Exception, e. It's just helpful
    # to know this for porting old scripts if you need to.


    except Exception as e:
        print(e)
