import os
import xlrd
import glob
import shutil

'''
healthySampleFolder = folder, ktory zostanie stworzony i w ktorym beda przechowywane zdjecia zdrowe
healthySamplesOutputPath = sciezka do pliku .txt zdrowego folderu (wybierasz dowolna nazwe byle byla w w.w folderze) 
 
sickSamplesFolder = analogicznie
sickSamplesOutputPath = analogicznie

sourceoOfImages = folder z plikami .jpg (byly dwie czesci w tym z kaggle'a , wiec wklej je do jednego folderu
excelSheet = nazwa pliku z danymi (musi być w formcie .xls) -- sama nazwa jesli w tym samym pliku co skrypt lub sciezka
'''



healthySamplesFolder = r'E:\PROGRAMS\GitHub\PROJEKT ZESPOLOWY\Excel\HealthySamples'
healthySamplesOutputPath = r'E:\PROGRAMS\GitHub\PROJEKT ZESPOLOWY\Excel\HealthySamples\healthyOutput.txt'

sickSamplesFolder = r'E:\PROGRAMS\GitHub\PROJEKT ZESPOLOWY\Excel\SickSamples'
sickSamplesOutputPath = r'E:\PROGRAMS\GitHub\PROJEKT ZESPOLOWY\Excel\SickSamples\sickOutput.txt'

sourceOfImages = r'E:\PROGRAMS\GitHub\PROJEKT ZESPOLOWY\img_part1'
excelSheet = "lesion_id.xls"

book = xlrd.open_workbook(excelSheet)
first_sheet = book.sheet_by_index(0)


if not os.path.exists(healthySamplesFolder):
    os.makedirs(healthySamplesFolder)

if not os.path.exists(sickSamplesFolder):
    os.makedirs(sickSamplesFolder)

sickSamplesOutputFile = open(sickSamplesOutputPath, "w")
healthySamplesOutputFile = open(healthySamplesOutputPath, "w")




for i in range(0, 10016):
    lesionType = first_sheet.cell(i, 2).value
    imageName = first_sheet.cell(i, 1).value
    if str(lesionType) == "nv":
        healthySamplesOutputFile.write(str(imageName + ".jpg" + "\n"))

    elif str(lesionType) in ['btl', 'df', 'akiec', 'bcc', 'mel']:
        sickSamplesOutputFile.write(str(imageName + ".jpg" + "\n"))

healthySamplesOutputFile.close()
sickSamplesOutputFile.close()

healthySamplesOutputFile = open(healthySamplesOutputPath, "r")
sickSamplesOutputFile = open(sickSamplesOutputPath, "r")

healthyLines = healthySamplesOutputFile.readlines()
sickLines = sickSamplesOutputFile.readlines()

for file in os.listdir(sourceOfImages):
    for healthyLine in healthyLines:
        if str(file + "\n") == str(healthyLine):
            shutil.copy(os.path.join(sourceOfImages, file), healthySamplesFolder)
    for sickLine in sickLines:
        if str(file + "\n") == str(sickLine):
            shutil.copy(os.path.join(sourceOfImages, file), sickSamplesFolder)

healthySamplesOutputFile.close()
sickSamplesOutputFile.close()
