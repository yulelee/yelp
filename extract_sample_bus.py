output = open('sample_bus.json', 'w')
data = open('yelp_academic_dataset_business.json')

line_number = 0

for line in data:
    output.write(line)
    line_number += 1
    if line_number >= 1000:
        break

output.close()
data.close()
