import csv 


with open("train.csv") as f_in, open("trainmodified.csv", 'w') as f_out:
    # Write header unchanged
    reader = csv.reader(f_in, delimiter=',')
    writer = csv.writer(f_out, delimiter=',')

    # Transform the rest of the lines
    for row in reader:
        
        if len(row[0]) < 31:
            id = row[0][20:(len(row[0])-4)]
            jpg = 'COCO_train2014_000000'+str(0)*(30-len(row[0]))+str(id)+'.jpg'
            txt = 'COCO_train2014_000000'+str(0)*(30-len(row[0]))+str(id)+'.txt'
            writer.writerow([jpg,txt ])
        else:
            writer.writerow([row[0],row[1]])