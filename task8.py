def take_feedback():
    relavent_images = input("Which images do you find relavent? (e.g. '1,4,5')\n")
    classifier = input("which classifier you want to use? (e.g. DT or SVM)\n")
    if classifier== 'DT':
        # call task 6 and return classifier
        pass
    else:
        # call task 7 and return classifier
        pass
index_file_path = input("Index file path\n")
query_image = input("Query Image Path\n")
t = input("t\n")
if index_file_path.startswith("va"):
    # call task 5

    classifier = take_feedback()

    # call task 5 again with 5*t
    # classify the result using above classifier and output top t images
    pass
else:
    # call task 4

    classifier = take_feedback()

    # call task 4 again with 5*t
    # classify the result using above classifier and output top t images
    pass

