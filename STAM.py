data  = np.zeros([10, 3, 2, 224, 224, 3])
data_dic  = {}
path  = "your/data/path"
for file in glob.glob(path):
    filename = file.split("/")[-1]  # get the name of the .jpg file
    img = np.asarray(Image.open(file))  # read the image as a numpy array

    # remove the alpha channel, normalize and subtract by 0.5:
    data_dic[filename] = (img[:, :, :3] / 255) - 0.5

Images = sorted([(key, value) for key, value in data_dic.items()])
Images = [value[1] for value in Images]
for num_of_t in range(data.shape[0]):
    for num_in_t in range(3):
        for direction in range(2):
            data[num_of_t, num_in_t, direction] = Images[num_of_t * 6 + num_in_t * 2 + direction]


print(f"CNNChannel Women's accuracy: {get_accuracy(CNNChannel_model, data, batch_size=25)}")
print(f"CNN Men's accuracy: {get_accuracy(CNN_model, data, batch_size=25)}")