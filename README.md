# CSE-515-Phase-3
feature_descriptor  {cm, elbp, hog}
image_type  {cc, con, emboss, jitter, neg, noise1, noise2, original, poster, rot, smooth, stipple} (According to the current dataset. We haven’t made any assumptions about the image type name and hence can differ based on the dataset.)
subject  [1-40] (string)
Sample ID  [1-10] (string)
Dimensionality reduction technique used - SVD
Classifier  {decision_tree, ppr, svm}
Task 1:
Command Syntax: python/python3 task1.py -fp [folder_path] -f [feature_descriptor] -k [latent_semantics_num] -qf [query images folder_path] -c [classifier]
Example: python task1.py -fp 'train_set/500' -f cm -k 10 -qf 'test_set/100' -c ppr
Task 2:
Command Syntax: python/python3 task2.py -fp [folder_path] -f [feature_descriptor] -k [latent_semantics_num] -qf [query images folder_path] -c [classifier]
Example: python task2.py -fp 'train_set/500' -f cm -k 10 -qf 'test_set/100' -c ppr
Task 3:
Command Syntax:  python/python3 task2.py -fp [folder_path] -f [feature_descriptor] -k [latent_semantics_num] -qf [query images folder_path] -c [classifier]
Example: python task2.py -fp 'train_set/500' -f cm -k 10 -qf 'test_set/100' -c ppr
   Task 4:
Command Syntax: python/python3 task4.py -fp [folder_path] -f [feature_descriptor] -l [number of layers] -k [latent_semantics_num] -kh [number of hash function per layer]
Example: python task4.py -fp “all" -f elbp -l 5 -k "all" -kh 10 
   Task 5:
Command Syntax: python/python3 task5.py -fp [folder_path] -f [feature_descriptor] -k [latent_semantics_num] -b [bits_per_dimension]
Example: python task5.py -fp '4000' -f cm -k -1 -b 3
Task 8:
Command Syntax: python/python3 task6_7_8.py
Then follow the command line instructions for input
Example:
python task6_7_8.py
500
index_lsh_cm_-1_5_9.json
500/image-cc-1-1.png
5
