import numpy as np
import numpy.linalg as la
from numpy.lib import math
class Personalised_Page_Rank:

    def __init__(self,topm = 10,types_of_labels=[]):
        self.topm=topm
        self.types_of_labels = types_of_labels
        # self.prediction_counter={}
        # for label_type in types_of_labels:
        #     self.prediction_counter[label_type]=0
        # pass

    def fit(self,features,labels):
        self.training_features=features
        self.training_labels = labels
        # self.seed_matrix = self.generate_seedMatrix(len(self.training_features),len(self.training_features)+1)

    def predict(self,test_images_features,test_images_labels):
        self.all_image_features = self.training_features
        print(len(self.training_features))
        self.predicted_labels=['']*len(test_images_labels)
        all_indices=np.arange(1,len(test_images_labels)+1,1)
        print("all indices type",type(all_indices))
        for each_test_image in test_images_features:
            self.all_image_features = np.append(self.all_image_features,[each_test_image],axis=0)
        
        for label_index in range(len(self.types_of_labels)-1):
            self.similarity_matrix = self.createSimilarityMatrix(self.all_image_features)
            self.transition_matrix = self.generate_transitionMatrix(self.similarity_matrix)
            input_images = [i for i in range(len(self.training_labels)) if self.training_labels[i]==self.types_of_labels[label_index]]
            self.seed_matrix = self.generate_seedMatrix(input_images,len(self.all_image_features))
            temp_page_rank = self.pageRank(self.transition_matrix,0.85,self.seed_matrix)
            test_page_rank = temp_page_rank[len(self.training_features):]
            print("len of test page rank",len(test_page_rank))
            numpyPage_Rank=np.array(test_page_rank)
            reshaped = numpyPage_Rank.reshape(numpyPage_Rank.shape[0],)
            
            reverse_indices = np.argsort(reshaped)
            print("reversed indices type",type(reverse_indices))
            getm=len(test_images_features)//len(self.types_of_labels)
            delete_indices = reverse_indices[::-1][:getm]
            print(delete_indices+len(self.training_features))
            print(len(self.all_image_features))
            for i in delete_indices:
                print(i,"type=",type(i), all_indices[i], len(self.predicted_labels))
                self.predicted_labels[all_indices[i]-1]=self.types_of_labels[label_index]
                # all_indices=np.delete(all_indices,i)
                # self.all_image_features=np.delete(self.all_image_features,(len(self.training_features)+i),axis=0)
            all_indices=np.delete(all_indices,[delete_indices])
            self.all_image_features=np.delete(self.all_image_features,[delete_indices+len(self.training_features)],axis=0)
            print("size of all images",len(self.all_image_features),"length of all indices",len(all_indices))
        print(all_indices)
        for x in all_indices:
            self.predicted_labels[x-1]=self.types_of_labels[len(self.types_of_labels)-1]





        # self.predicted_labels=[]
        # self.all_image_features = self.training_features 
        # for each_test_image in test_images_features:
        #     self.all_image_features = np.append(self.all_image_features,[each_test_image],axis=0)
        #     self.similarity_matrix = self.createSimilarityMatrix(self.all_image_features)
        #     self.transition_matrix = self.generate_transitionMatrix(self.similarity_matrix)
        #     output=[]
        #     for label in self.types_of_labels:
        #         input_images = [i for i in range(len(self.training_labels)) if self.training_labels[i]==label]
        #         self.seed_matrix = self.generate_seedMatrix(input_images,len(self.all_image_features))
        #         output.append([label,self.pageRank(self.transition_matrix,0.85,self.seed_matrix)])
        #     compare=[]
        #     for item in output:
        #         compare.append([item[0],len(self.training_features),item[1][len(self.training_features)]])
        #     self.predicted_labels.append(sorted(compare,reverse=True,key = lambda x:x[2])[0][0])
	        


        # print("predicted images length",len(self.predicted_labels),"test_images_labels length",len(test_images_labels))
        # for j in range(len(test_images_features)):
        #     print(j,"->",self.predicted_labels[j],"original->",test_images_labels[j])
        
        # self.similarity_matrix = self.createSimilarityMatrix(self.all_image_features)
        # self.transition_matrix = self.generate_transitionMatrix(self.similarity_matrix)
	    
	    

        # output=[]
        # for label in self.types_of_labels:
	    #     #calculating page rank for that label

        #     input_images = [i for i in range(len(self.training_labels)) if self.training_labels[i]==label]
        #     self.seed_matrix = self.generate_seedMatrix(input_images,len(self.all_image_features),len(test_images_features))
        #     output.append([label,self.pageRank(self.transition_matrix,0.15,self.seed_matrix)])
	
        # self.predicted_labels=[]
        # for i in range(len(self.training_features),len(output[0][1])):
	    #     compare=[]
	    #     for item in output:
		#         compare.append([item[0],i,item[1][i]])
	    #     self.predicted_labels.append(sorted(compare,reverse=True,key = lambda x:x[2])[0][0])

        # print("predicted images length",len(self.predicted_labels),"test_images_labels length",len(test_images_labels))
        # for j in range(len(test_images_features)):
        #     print(j,"->",self.predicted_labels[j],"original->",test_images_labels[j])


        # self.all_image_features = self.training_features 
        # #print(np.shape(self.all_image_features))
        # #print("type of training label",type(self.training_labels),"size is",np.shape(self.training_labels))
        # #print("type of image features",type(self.all_image_features))
        # self.test_predicted_dict={}
        # test_id =0
        # #print("type of test images",type(test_images_features))
        # #print("shape of test images ",len(test_images_labels))
        # for each_test_query in test_images_features:
        #     #print("shape of 1 test image ",type(each_test_query),"size of 1 test image is",len(each_test_query))
        #     self.all_image_features=np.append(self.all_image_features,[each_test_query],axis=0)
        #     #print("after adding test image",len(self.all_image_features))
        #     self.similarity_matrix = self.createSimilarityMatrix(self.all_image_features)


        #     rows,cols = np.shape(self.similarity_matrix)
        #     n = 400
        #     topn = np.zeros((np.shape(self.similarity_matrix)[0],n))
        #     for i in range(rows):
        #         temp=[]
        #         temp = np.argpartition(self.similarity_matrix[i],-n)[-n:]
        #         for j in range(n):
        #             topn[i][j] = temp[j]
        #         topn[i].sort()
        #     temp = np.zeros(np.shape(self.similarity_matrix))
        #     for i in range(len(topn)):
        #         for j in range(len(topn[i])):
        #             temp[i][int(topn[i][j])] = self.similarity_matrix[i][int(topn[i][j])]






        #     self.transition_matrix = self.generate_transitionMatrix(temp)
        #     # self.seed_matrix = self.generate_seedMatrix(len(self.all_image_features)-1,self.transition_matrix.shape[0])
        #     self.page_rank = self.pageRank(self.transition_matrix,0.85,self.seed_matrix)

        #     numpyPage_Rank=np.array(self.page_rank)
        #     numpyPage_Rank = numpyPage_Rank[:-1]
        #     #np.delete(numpyPage_Rank,len(numpyPage_Rank)-1,axis=0)
        #     #print("numpyPageRank shape",numpyPage_Rank.shape)
        #     reshaped = numpyPage_Rank.reshape(numpyPage_Rank.shape[0],)
        #     #print(sorted(numpyPage_Rank))

        #     #print("reshaped",reshaped)
        #     sorted_indices = reversed(np.argsort(reshaped))
        #     #print(sorted_indices)

        #     for key in self.prediction_counter.keys():
        #         self.prediction_counter[key]=0
        #     getm=0
        #     for i in sorted_indices:
        #         # print("pagerank=",numpyPage_Rank[i],"in sortedindices",i,"label =",self.training_labels[i])
        #         self.prediction_counter[self.training_labels[i]]+=1
        #         getm+=1
        #         if getm>self.topm:
        #             break
        #     #print("value of getm",getm)
        #     #self.reverse_prediction_counter = sorted(self.prediction_counter.items(),key = lambda x:x[1], reverse=True)
        #     # print(type(self.reverse_prediction_counter))
        #     # for key in self.reverse_prediction_counter:
        #     #     print(key)
        #     # for key in self.reverse_prediction_counter.keys():
        #     #     self.test_predicted_dict[each_test_query]=key
        #     #     break;
        #     # self.test_predicted_dict[each_test_query] = self.reverse_prediction_counter[0][0]
        #     max_label = max(self.prediction_counter, key = self.prediction_counter.get)
        #     print("test_id",test_id,"->",max_label,"original label",test_images_labels[test_id])
        #     self.test_predicted_dict[test_id] = max_label
        #     test_id+=1

        #     self.all_image_features = self.all_image_features[:-1]
            
            #np.delete(self.all_image_features,len(self.all_image_features)-1,axis=0)
        return self.predicted_labels

    def accuracy(self,test_predicted_labels,test_images_labels):
        count=0
        # for key, val in test_predicted_dict.items():
        #     if val==test_images_labels[key]:
        #         count+=1
        print("predicted length=",len(test_predicted_labels),"length of test labels=",len(test_images_labels))
        print(test_predicted_labels)
        print(test_images_labels)
        for i in range(len(test_predicted_labels)):
            print(test_predicted_labels[i],"->",test_images_labels[i])
            if test_predicted_labels[i] == test_images_labels[i]:
                count+=1
        print("count",count)
        print("len of test labels", len(test_images_labels))
        return (count*100)/len(test_images_labels)



    def pageRank(self,linkMatrix,d,seedMatrix):
        n = linkMatrix.shape[0]
        r=100 * np.ones((n,1)) / n
        last = r
        r = d*np.matmul(linkMatrix,r) + (1-d) * seedMatrix
        while la.norm(last-r) > 0.01:
            last =r
            r = d*np.matmul(linkMatrix,last) + (1-d)* seedMatrix

        
        return r

    def generate_transitionMatrix(self,subject_subject_matrix):
        subject_subject_matrix_transpose = np.transpose(subject_subject_matrix)
        answer_matrix = np.zeros((subject_subject_matrix_transpose.shape[0],subject_subject_matrix_transpose.shape[1]))
        for j in np.arange(subject_subject_matrix_transpose.shape[0]):
            columnSum = np.sum(np.absolute(subject_subject_matrix_transpose[j]))
            for i in np.arange(subject_subject_matrix_transpose.shape[1]):
                if columnSum==0:
                    answer_matrix[j][i] = subject_subject_matrix_transpose[j][i]
                    continue
                answer_matrix[j][i] = subject_subject_matrix_transpose[j][i]/columnSum
        
        return answer_matrix.transpose()


    def generate_seedMatrix(self,indexes,length):
        seedmatrix = np.zeros((length,1))
        # mag_seed = len(indexes)+length_test_images
        # for i in indexes:
        #     seedmatrix[i]=1/mag_seed
        # for i in range(length_test_images):
        #     seedmatrix[len(self.training_features)+i]=1/mag_seed
        # # seedmatrix[index]=1/length
        for i in indexes:
            seedmatrix[i]=1/(len(indexes))
        
        return seedmatrix

    def createSimilarityMatrix(self,features):
        image_similarity_matrix=[]
        for row_img_feature in features:
            one_img_similarity=[]
            for col_img_feature in features:
                one_img_similarity.append(np.dot(row_img_feature,col_img_feature.T))
            image_similarity_matrix.append(one_img_similarity)
        # for i in range(features.shape[0]):
        #     one_img_similarity=[]
        #     for i in range(features.shape[0]):
        #         one_img_similarity.append(features[i].dot(features[i].transpose()))
        #     image_similarity_matrix.append(one_img_similarity)
        return image_similarity_matrix
        

