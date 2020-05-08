#holidays bavaria 2017 and 2018
holidays = [
    "01.01.2017", "06.01.2017", "14.04.2017", "17.04.2017", "17.04.2017", "01.05.2017", "25.05.2017",
    "05.06.2017", "15.06.2017", "15.08.2017", "03.10.2017", "31.10.2017", "01.11.2017", "25.12.2017",
    "26.12.2017",
    "01.01.2018", "06.01.2018", "30.03.2018", "02.04.2018", "01.05.2018", "10.05.2018",
    "21.05.2018", "31.05.2018", "15.08.2018", "03.10.2018", "01.11.2018", "25.12.2018", "26.12.2018",
]
"""
df = df.assign(Montag=lambda x: x.index.weekday == 0,
               Dienstag=lambda x: x.index.weekday == 1,
               Mittwoch=lambda x: x.index.weekday == 2,
               Donnerstag=lambda x: x.index.weekday == 3,
               Freitag=lambda x: x.index.weekday == 4,
               Samstag=lambda x: x.index.weekday == 5,
               Sonntag=lambda x: x.index.weekday == 6)
"""

def label_holidays(row):
    if row.name.strftime("%d.%m.%Y") in holidays:
        return True
    else:
        return False


