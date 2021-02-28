# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:46:18 2020
@author: Ahmad Al Musawi
"""


def start():
    print("welcome to Group Detection Model")
    path = 'D:\Documents\Research Projects\Complex Networks Researches\Groups Detection Model in Online Examination\Coding\Dataset'
    datasets = ["\cg.txt", "\ml.txt", "\ds.txt"]
    print(datasets)
    d = int(input("Enter index of file"))
    newPath = path + datasets[d]

    lines = open(newPath, "r", encoding=('utf-8')).readlines()
    # print(lines)
    text = ""
    for line in lines:
        text +=line
        
    anss = text.split('***')
    answers = []
    
    r=1
    for answer in anss:
        # print(r,'- ', "*****************************************")
        ans = answer.split('\t')
        answers.append(ans)
        r+=1
    ans7 = []
    students = []
    for i in range(len(answers[0])):
        print(i, answers[0][i])
    question = int(input("Enter Question index to continue"))
    
    print(answers[0][question])
    for i in range(1,len(answers)-1):
        ans7.append(answers[i][question])
        students.append(answers[i][0])
    ss =  My_Clustering(ans7, students).to_numpy()
    for i in ss:
        print (i[1])
    
    # results = []
    # for i in range(len(answers)-1):
    #     res = []
    #     for j in range(len(answers)-1):
    #         b = answers[i][7]
    #         a = answers[j][7]
    #         res.append(similar(a,b))
    #     results.append(res)
    # for i in results:
    #     line = ""
    #     for j in i:
    #         line = line + "\t" + str(j)
    #     print(line)


def My_Clustering(answers, students):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words={'english'})
    X = vectorizer.fit_transform(answers)
    # ---------------------------------------------
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    Sum_of_squared_distances = []
    K = range(2,10)
    for k in K:
       km = KMeans(n_clusters=k, max_iter=200, n_init=10)
       km = km.fit(X)
       Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    # ---------------------------------------------
    import pandas as pd
    true_k = int(input("Enter Number of clusters"))
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
    model.fit(X)
    labels=model.labels_
    student_cl=pd.DataFrame(list(zip(students,labels)),columns=['student','cluster'])
    # print(student_cl.sort_values(by=['cluster']))
    return student_cl

    # ---------------------------------------------
    # return 
    # from wordcloud import WordCloud
    # result={'cluster':labels,'ans':answers}
    # result=pd.DataFrame(result)
    # for k in range(0,true_k):
    #    s=result[result.cluster==k]
    #    text=s['ans'].str.cat(sep=' ')
    #    text=text.lower()
    #    text=' '.join([word for word in text.split()])
    #    # wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    #    print('Cluster: {}'.format(k))
    #    print('Students')
    #    titles=wiki_cl[wiki_cl.cluster==k]['title']         
    #    print(titles.to_string(index=False))
    #    plt.figure()
    #    # plt.imshow(wordcloud, interpolation="bilinear")
    #    plt.axis("off")
    #    plt.show()

def result_clustering():
    import numpy as np
    import matplotlib.pyplot as plt
    print("welcome to Group Detection Model")
    path = 'D:\Documents\Research Projects\Complex Networks Researches\Groups Detection Model in Online Examination\Coding\Dataset'
    datasets = ["\cg result.txt", "\ml result.txt", "\ds result.txt"]
    print(datasets)
    d = int(input("Enter index of file"))
    newPath = path + datasets[d]
    
    lines = open(newPath, "r", encoding=('utf-8')).readlines()
    # print(lines)
    text = ""
    for line in lines:
        text +=line.strip("\n")
    # print(text)
    anss = text.split('***')
    sheet= []
    for i in range(0,len(anss)-1):
        answer = anss[i]
        ans = answer.split('\t')
        sheet.append(ans[1:-1])  
    print(sheet[0])
    input("Press any key")
    X = np.array(sheet[1:])
    plt.scatter(X[:, 0], X[:, 1], s=50);
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5, random_state=0, max_iter = 300).fit(X)
    Y = kmeans.fit_predict(X)
    print(type(X), type(Y))
    for i in range(len(X)):
        print(Y[i])
    plt.scatter(X[:, 0], X[:, 2], c=Y, s=50, cmap='viridis')    
    # centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 2], c='black', s=200, alpha=0.5);    
    
    # plt.scatter(X[:,0], X[:, 1], s=50, cmap='viridis')
    # return Y


# def words_bag(text):
#     print(text)

# from difflib import SequenceMatcher

# def similar(a, b):
#     return SequenceMatcher(None, a, b).ratio()

# def longestSubstringFinder(string1, string2):
#     answer = ""
#     len1, len2 = len(string1), len(string2)
#     for i in range(len1):
#         match = ""
#         for j in range(len2):
#             if (i + j < len1 and string1[i + j] == string2[j]):
#                 match += string2[j]
#             else:
#                 if (len(match) > len(answer)): answer = match
#                 match = ""
#     return answer