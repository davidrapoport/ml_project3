import numpy as np
import csv


# Load all training inputs to a python list
train_inputs = []
with open('data/train_inputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for train_input in reader: 
        train_input_no_id = []
        for dimension in train_input[1:]:
            train_input_no_id.append(float(dimension))
        train_inputs.append(np.asarray(train_input_no_id)) # Load each sample as a numpy array, which is appened to the python list

# Load all training ouputs to a python list
train_outputs = []
with open('data/train_outputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for train_output in reader:  
        train_output_no_id = int(train_output[1])
        train_outputs.append(train_output_no_id)

test_inputs = []
with open('data/test_inputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for test_input in reader: 
        test_input_no_id = []
        for dimension in test_input[1:]:
            test_input_no_id.append(float(dimension))
        test_inputs.append(np.asarray(test_input_no_id)) # Load each sample as a numpy array, which is appened to the python list


# Convert python lists to numpy arrays
train_inputs_np = np.asarray(train_inputs)
train_outputs_np = np.asarray(train_outputs)
test_inputs_np = np.asarray(test_inputs)

# Save as numpy array files
np.save('data/train_inputs', train_inputs_np)
np.save('data/train_outputs', train_outputs_np)
np.save('data/test_inputs', test_inputs_np)


# Following code courtesy of http://pjreddie.com/projects/mnist-in-csv/
# Dataset courtesy of http://yann.lecun.com/exdb/mnist/
def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf+"inputs.csv", "w")
    o2 = open(outf+"outputs.csv", "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []
    images_np = []
    labels = []

    for i in range(n):
        labels.append(ord(l.read(1)))
        image = []
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)
        images_np.append(np.asarray(image))

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    o2.write("\n\r".join([str(label) for label in labels]))
    f.close()
    o.close()
    o2.close()
    l.close()
    mnist_train_inputs = np.asarray(images_np)
    mnist_train_outputs = np.asarray(labels)
    np.save(outf+"inputs", mnist_train_inputs)
    np.save(outf+"outputs", mnist_train_outputs)

convert("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
        "data/mnist_train_extra_", 60000)


